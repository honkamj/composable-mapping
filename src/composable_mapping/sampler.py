"""Interpolation class wrappers"""

from abc import abstractmethod
from math import ceil, floor
from typing import Callable, List, Optional, Sequence, Tuple, Union, cast

from numpy import argsort
from torch import Tensor
from torch import any as torch_any
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import linspace, long, ones, ones_like, tensor, zeros
from torch.nn.functional import conv1d, conv_transpose1d

from composable_mapping.mappable_tensor.affine_transformation import (
    DiagonalAffineTransformation,
    HostAffineTransformation,
    HostDiagonalAffineTransformation,
    IdentityAffineTransformation,
    IHostAffineTransformation,
)
from composable_mapping.mappable_tensor.mappable_tensor import (
    MappableTensor,
    PlainTensor,
)
from composable_mapping.util import (
    avg_pool_nd_function,
    crop_and_then_pad_spatial,
    get_batch_dims,
    get_batch_shape,
    get_channels_shape,
    get_n_channel_dims,
    get_spatial_dims,
    get_spatial_shape,
    includes_padding,
    is_croppable_first,
    split_shape,
)

from .dense_deformation import generate_voxel_coordinate_grid, interpolate
from .interface import ISampler


class _BaseSeparableSampler(ISampler):
    """Base sampler in voxel coordinates which can be implemented as a
    separable convolution"""

    def __init__(
        self,
        extrapolation_mode: str,
        mask_extrapolated_regions_for_empty_volume_mask: bool,
        convolution_threshold: float,
        mask_threshold: float,
        interpolating_sampler: bool,
    ) -> None:
        if extrapolation_mode not in ("zeros", "border", "reflection"):
            raise ValueError("Unknown extrapolation mode")
        self._extrapolation_mode = extrapolation_mode
        self._mask_extrapolated_regions_for_empty_volume_mask = (
            mask_extrapolated_regions_for_empty_volume_mask
        )
        self._convolution_threshold = convolution_threshold
        self._mask_threshold = mask_threshold
        self._interpolating_sampler = interpolating_sampler

    @property
    @abstractmethod
    def _kernel_size(self) -> Tuple[float, bool, bool]:
        """Return the kernel size for the convolution and whether min and max
        are inclusive or not"""

    @abstractmethod
    def _evaluate_interpolation_kernel(self, coordinates: Tensor) -> Tensor:
        """Return the interpolation kernel for the given 1d coordinates"""

    def __call__(self, volume: MappableTensor, coordinates: MappableTensor) -> MappableTensor:
        if coordinates.n_channel_dims != 1:
            raise ValueError("Interpolation assumes single channel coordinates")
        if coordinates.channels_shape[0] != len(volume.spatial_shape):
            raise ValueError("Interpolation assumes same number of channels as spatial dims")
        interpolated = self._interpolate_conv(volume, coordinates)
        if interpolated is None:
            return self._interpolate_grid_sample(volume, coordinates)
        return interpolated

    @property
    def _padding_mode_and_value(self) -> Tuple[str, float]:
        return {
            "zeros": ("constant", 0.0),
            "border": ("replicate", 0.0),
            "reflection": ("reflect", 0.0),
        }[self._extrapolation_mode]

    @staticmethod
    def _obtain_flipping_permutation(
        matrix: Tensor,
    ) -> Optional["_FlippingPermutation"]:
        largest_indices = matrix[:-1, :-1].abs().argmax(dim=1).tolist()
        n_dims = matrix.shape[-1] - 1
        if not _FlippingPermutation.is_valid_permutation(n_dims, largest_indices):
            return None
        permutation = tuple(argsort(largest_indices))
        flipped_spatial_dims = [
            largest_index
            for column, largest_index in enumerate(largest_indices)
            if matrix[largest_index, column] < 0
        ]
        return _FlippingPermutation(permutation, flipped_spatial_dims)

    def _extract_conv_interpolatable_parameters(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, "_FlippingPermutation"]]:
        if voxel_coordinates.displacements is not None:
            return None
        grid = voxel_coordinates.grid
        assert grid is not None
        affine_transformation = grid.affine_transformation
        channels_shape = affine_transformation.channels_shape
        if channels_shape[0] != channels_shape[1]:
            return None
        n_dims = len(grid.spatial_shape)
        host_matrix = grid.affine_transformation.as_host_matrix()
        if host_matrix is None:
            return None
        host_matrix = host_matrix.view(-1, n_dims + 1, n_dims + 1)
        flipping_permutation = self._obtain_flipping_permutation(host_matrix[0])
        if flipping_permutation is None:
            return None
        affine_transformation = (
            flipping_permutation.as_transformation(spatial_shape=volume.spatial_shape)
            @ affine_transformation
        )
        host_matrix = affine_transformation.as_host_matrix()
        assert host_matrix is not None
        host_matrix = host_matrix.view(-1, n_dims + 1, n_dims + 1)
        diagonal = host_matrix[0, :-1, :-1].diagonal()
        translation = host_matrix[0, :-1, -1].clone()
        exact_transformation = affine_transformation.cast(device=torch_device("cpu"))
        transposed_convolve = diagonal < 0.75
        if (diagonal == 0.0).any():
            return None
        downsampling_factor = (
            diagonal.round() * (~transposed_convolve) + (1 / diagonal).round() * transposed_convolve
        )
        downsampling_factor[transposed_convolve] = 1 / downsampling_factor[transposed_convolve]
        rounded_diagonal_transformation = DiagonalAffineTransformation(
            diagonal=downsampling_factor, translation=translation
        )
        shape_tensor = tensor(grid.spatial_shape, device=torch_device("cpu"), dtype=grid.dtype)
        shape_transformation = DiagonalAffineTransformation(diagonal=shape_tensor)
        unit_grid = generate_voxel_coordinate_grid(
            shape=(2,) * n_dims, device=torch_device("cpu"), dtype=grid.dtype
        )
        corner_points = (exact_transformation @ shape_transformation)(unit_grid)
        rounded_diagonal_corner_points = (  # pylint: disable=not-callable
            rounded_diagonal_transformation @ shape_transformation
        )(unit_grid)
        max_rounded_diagonal_displacements = (
            (corner_points - rounded_diagonal_corner_points)
            .abs()
            .amax(
                dim=get_batch_dims(unit_grid.ndim, n_channel_dims=1)
                + get_spatial_dims(unit_grid.ndim, n_channel_dims=1)
            )
        )
        if torch_any(max_rounded_diagonal_displacements > self._convolution_threshold):
            return None
        if self._interpolating_sampler:
            rounded_translation = translation.round()
            rounded_diagonal_and_translation_transformation = DiagonalAffineTransformation(
                diagonal=downsampling_factor, translation=rounded_translation
            )
            rounded_diagonal_and_translation_corner_points = (  # pylint: disable=not-callable
                rounded_diagonal_and_translation_transformation @ shape_transformation
            )(unit_grid)
            convolve = (
                (corner_points - rounded_diagonal_and_translation_corner_points)
                .abs()
                .amax(
                    dim=get_batch_dims(unit_grid.ndim, n_channel_dims=1)
                    + get_spatial_dims(unit_grid.ndim, n_channel_dims=1)
                )
                > self._convolution_threshold
            ) & (~transposed_convolve)
            no_convolve_or_transposed_convolve = (~transposed_convolve) & (~convolve)
            translation[no_convolve_or_transposed_convolve] = translation[
                no_convolve_or_transposed_convolve
            ].round()
        else:
            convolve = ~transposed_convolve
        return downsampling_factor, translation, convolve, transposed_convolve, flipping_permutation

    def _obtain_conv_parameters(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> Optional[
        Tuple[
            Sequence[Optional[Tensor]],
            Sequence[int],
            Sequence[Tuple[int, int]],
            Sequence[Tuple[int, int]],
            Sequence[bool],
            "_FlippingPermutation",
        ]
    ]:
        conv_interpolation_parameters = self._extract_conv_interpolatable_parameters(
            volume, voxel_coordinates
        )
        if conv_interpolation_parameters is None:
            return None
        downsampling_factor, translation, convolve, transposed_convolve, flipping_permutation = (
            conv_interpolation_parameters
        )
        pads_or_crops: List[Tuple[int, int]] = []
        post_pads_or_crops: List[Tuple[int, int]] = []
        kernels: List[Optional[Tensor]] = []
        (kernel_width, inclusive_min, inclusive_max) = self._kernel_size
        for (
            dim_size_volume,
            dim_size_grid,
            dim_convolve,
            dim_transposed_convolve,
            dim_translation,
            dim_downsampling_factor,
        ) in zip(
            flipping_permutation.permute_sequence(volume.spatial_shape),
            voxel_coordinates.spatial_shape,
            convolve.tolist(),
            transposed_convolve.tolist(),
            translation.tolist(),
            downsampling_factor.tolist(),
        ):
            lower_flooring_function = self._get_flooring_function(inclusive_min)
            upper_flooring_function = self._get_flooring_function(inclusive_max)
            min_coordinate = dim_translation
            max_coordinate = dim_translation + dim_downsampling_factor * (dim_size_grid - 1)
            if dim_convolve or dim_transposed_convolve:
                padding_lower = lower_flooring_function(kernel_width / 2 - dim_translation)
                padding_upper = upper_flooring_function(
                    kernel_width / 2
                    + dim_translation
                    + dim_downsampling_factor * (dim_size_grid - 1)
                    - (dim_size_volume - 1)
                )
                if dim_transposed_convolve:
                    start_kernel_coordinate = (
                        1
                        - dim_translation
                        + upper_flooring_function(
                            (kernel_width / 2 - (1 - dim_translation)) / dim_downsampling_factor
                        )
                        * dim_downsampling_factor
                    )
                    end_kernel_coordinate = (
                        -dim_translation
                        - lower_flooring_function(
                            (kernel_width / 2 - dim_translation) / dim_downsampling_factor
                        )
                        * dim_downsampling_factor
                    )
                    kernel_step_size = dim_downsampling_factor
                else:
                    relative_coordinate = dim_translation - floor(dim_translation)
                    start_kernel_coordinate = (
                        -lower_flooring_function(kernel_width / 2 - relative_coordinate)
                        - relative_coordinate
                    )
                    end_kernel_coordinate = upper_flooring_function(
                        kernel_width / 2 - (1 - relative_coordinate)
                    ) + (1 - relative_coordinate)
                    kernel_step_size = 1
                kernel_coordinates = linspace(
                    start_kernel_coordinate,
                    end_kernel_coordinate,
                    int(
                        round(
                            abs(end_kernel_coordinate - start_kernel_coordinate) / kernel_step_size
                        )
                    )
                    + 1,
                    dtype=voxel_coordinates.dtype,
                    device=voxel_coordinates.device,
                )
                kernel = self._evaluate_interpolation_kernel(kernel_coordinates)
            else:
                kernel = None
                padding_lower = -int(min_coordinate)
                padding_upper = int(max_coordinate) - (dim_size_volume - 1)
            post_pad_or_crop_lower = (
                -int(
                    round(
                        (min_coordinate + padding_lower + start_kernel_coordinate)
                        / dim_downsampling_factor
                    )
                )
                if dim_transposed_convolve
                else 0
            )
            post_pad_or_crop_upper = (
                -int(
                    round(
                        (
                            (dim_size_volume - 1)
                            + padding_upper
                            - end_kernel_coordinate
                            - max_coordinate
                        )
                        / dim_downsampling_factor
                    )
                )
                if dim_transposed_convolve
                else 0
            )
            pads_or_crops.append((padding_lower, padding_upper))
            kernels.append(kernel)
            post_pads_or_crops.append((post_pad_or_crop_lower, post_pad_or_crop_upper))
        padding_mode, _padding_value = self._padding_mode_and_value
        if not is_croppable_first(
            spatial_shape=volume.spatial_shape, pads_or_crops=pads_or_crops, mode=padding_mode
        ):
            return None
        strides = (
            (
                downsampling_factor * (~transposed_convolve)
                + (1 / downsampling_factor) * transposed_convolve
            )
            .round()
            .to(dtype=long)
        )
        return (
            kernels,
            strides.tolist(),
            pads_or_crops,
            post_pads_or_crops,
            transposed_convolve.tolist(),
            flipping_permutation,
        )

    def _interpolate_conv(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> Optional[MappableTensor]:
        conv_parameters = self._obtain_conv_parameters(volume, voxel_coordinates)
        if conv_parameters is None:
            return None
        (
            kernels,
            strides,
            pads_or_crops,
            post_pads_or_crops,
            transposed_convolve,
            flipping_permutation,
        ) = conv_parameters

        values, mask = volume.generate(
            generate_missing_mask=includes_padding(pads_or_crops)
            or self._mask_extrapolated_regions_for_empty_volume_mask,
            cast_mask=False,
        )
        values = flipping_permutation(values, n_channel_dims=volume.n_channel_dims)
        padding_mode, padding_value = self._padding_mode_and_value
        interpolated_values = crop_and_then_pad_spatial(
            values,
            pads_or_crops=pads_or_crops,
            mode=padding_mode,
            value=padding_value,
            n_channel_dims=volume.n_channel_dims,
        )
        interpolated_values = self._separable_conv_nd(
            volume=interpolated_values,
            kernels=kernels,
            strides=strides,
            transposed=transposed_convolve,
            post_pads_or_crops=post_pads_or_crops,
            n_channel_dims=volume.n_channel_dims,
        )
        if mask is None:
            interpolated_mask: Optional[Tensor] = None
        else:
            mask = flipping_permutation(mask, n_channel_dims=volume.n_channel_dims)
            interpolated_mask = crop_and_then_pad_spatial(
                mask,
                pads_or_crops=pads_or_crops,
                mode="constant",
                value=False,
                n_channel_dims=volume.n_channel_dims,
            )
            interpolated_mask = (
                self._separable_conv_nd(
                    volume=(~interpolated_mask).to(dtype=voxel_coordinates.dtype),
                    kernels=[None if kernel is None else kernel.abs() for kernel in kernels],
                    strides=strides,
                    transposed=transposed_convolve,
                    post_pads_or_crops=post_pads_or_crops,
                    n_channel_dims=volume.n_channel_dims,
                )
                <= self._mask_threshold
            )
        return PlainTensor(
            interpolated_values, interpolated_mask, n_channel_dims=volume.n_channel_dims
        )

    def _separable_conv_nd(
        self,
        volume: Tensor,
        kernels: Sequence[Optional[Tensor]],
        strides: Sequence[int],
        transposed: Sequence[bool],
        post_pads_or_crops: Sequence[Tuple[int, int]],
        n_channel_dims: int,
    ) -> Tensor:
        post_pads_or_crops = list(post_pads_or_crops)
        if (
            len(kernels) != len(get_spatial_dims(volume.ndim, n_channel_dims))
            or len(kernels) != len(strides)
            or len(kernels) != len(transposed)
            or len(kernels) != len(post_pads_or_crops)
        ):
            raise ValueError(
                "Invalid number of kernels, strides, transposed, or post_pads_or_crops"
            )
        for spatial_dim, (kernel, stride, dim_transposed, post_padding_or_crop) in enumerate(
            zip(kernels, strides, transposed, post_pads_or_crops)
        ):
            if kernel is None or kernel.size(0) == 1:
                assert dim_transposed is False
                dim = get_spatial_dims(volume.ndim, n_channel_dims)[spatial_dim]
                volume = volume[(slice(None),) * dim + (slice(None, None, stride),)]
                if kernel is not None:
                    volume = kernel * volume
            else:
                if dim_transposed:
                    shared_crop = max(-max(post_padding_or_crop), 0)
                    if shared_crop > 0:
                        post_pads_or_crops[spatial_dim] = (
                            post_padding_or_crop[0] + shared_crop,
                            post_padding_or_crop[1] + shared_crop,
                        )
                volume = self._conv1d(
                    volume,
                    spatial_dim=spatial_dim,
                    kernel=kernel,
                    stride=stride,
                    n_channel_dims=n_channel_dims,
                    transposed=dim_transposed,
                    padding=shared_crop if dim_transposed else 0,
                )
        padding_mode, padding_value = self._padding_mode_and_value
        volume = crop_and_then_pad_spatial(
            volume,
            pads_or_crops=post_pads_or_crops,
            mode=padding_mode,
            value=padding_value,
            n_channel_dims=n_channel_dims,
        )
        return volume

    @staticmethod
    def _conv1d(
        volume: Tensor,
        spatial_dim: int,
        kernel: Tensor,
        stride: int,
        n_channel_dims: int,
        transposed: bool,
        padding: int,
    ) -> Tensor:
        dim = get_spatial_dims(volume.ndim, n_channel_dims)[spatial_dim]
        volume = volume.moveaxis(dim, -1)
        dim_excluded_shape = volume.shape[:-1]
        volume = volume.reshape(-1, 1, volume.size(-1))
        conv_function = cast(Callable[..., Tensor], conv1d if not transposed else conv_transpose1d)
        convolved = conv_function(  # pylint: disable=not-callable
            volume,
            kernel[None, None],
            bias=None,
            stride=(stride,),
            padding=padding,
        ).reshape(dim_excluded_shape + (-1,))
        return convolved.moveaxis(-1, dim)

    @staticmethod
    def _get_flooring_function(inclusive: bool) -> Callable[[Union[float, int]], int]:
        if inclusive:
            return lambda x: int(floor(x))
        return lambda x: int(ceil(x - 1))

    def _interpolate_grid_sample(
        self, volume: MappableTensor, voxel_coordinates: MappableTensor
    ) -> MappableTensor:
        volume_values, volume_mask = volume.generate(
            generate_missing_mask=self._mask_extrapolated_regions_for_empty_volume_mask,
            cast_mask=False,
        )
        coordinate_values = voxel_coordinates.generate_values()
        interpolated_values = self.sample_values(volume_values, coordinate_values)
        if volume_mask is not None:
            interpolated_mask: Optional[Tensor] = self.sample_mask(
                volume_mask,
                coordinate_values,
            )
        else:
            interpolated_mask = None
        return PlainTensor(
            interpolated_values, interpolated_mask, n_channel_dims=volume.n_channel_dims
        )


class LinearInterpolator(_BaseSeparableSampler):
    """Linear interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-5,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            interpolating_sampler=True,
        )

    def sample_values(
        self,
        values: Tensor,
        voxel_coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            values,
            voxel_coordinates,
            mode="bilinear",
            padding_mode=self._extrapolation_mode,
        )

    @property
    def _kernel_size(self) -> Tuple[float, bool, bool]:
        return (2.0, False, False)

    def _evaluate_interpolation_kernel(self, coordinates: Tensor) -> Tensor:
        return 1 - coordinates.abs()

    def sample_mask(
        self,
        mask: Tensor,
        voxel_coordinates: Tensor,
    ) -> Tensor:
        interpolated_mask = interpolate(
            volume=mask.to(voxel_coordinates.dtype),
            grid=voxel_coordinates,
            mode="bilinear",
            padding_mode="zeros",
        )
        return interpolated_mask >= 1 - self._mask_threshold


class NearestInterpolator(_BaseSeparableSampler):
    """Nearest neighbour interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-5,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            interpolating_sampler=True,
        )

    def sample_values(
        self,
        values: Tensor,
        voxel_coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            values,
            voxel_coordinates,
            mode="nearest",
            padding_mode=self._extrapolation_mode,
        )

    def sample_mask(
        self,
        mask: Tensor,
        voxel_coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            volume=mask.to(voxel_coordinates.dtype),
            grid=voxel_coordinates,
            mode="nearest",
            padding_mode="zeros",
        ).to(mask.dtype)

    @property
    def _kernel_size(self) -> Tuple[float, bool, bool]:
        return (1.0, True, False)

    def _evaluate_interpolation_kernel(self, coordinates: Tensor) -> Tensor:
        return ones_like(coordinates)


class BicubicInterpolator(_BaseSeparableSampler):
    """Bicubic interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-5,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            interpolating_sampler=True,
        )
        self._mask_threshold = mask_threshold

    def sample_values(
        self,
        values: Tensor,
        voxel_coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            values,
            voxel_coordinates,
            mode="bicubic",
            padding_mode=self._extrapolation_mode,
        )

    def sample_mask(
        self,
        mask: Tensor,
        voxel_coordinates: Tensor,
    ) -> Tensor:
        n_spatial_dims = get_channels_shape(voxel_coordinates.shape, n_channel_dims=1)[0]
        n_channel_dims = get_n_channel_dims(mask.ndim, n_spatial_dims)
        batch_shape, channels_shape, spatial_shape = split_shape(
            mask.shape, n_channel_dims=n_channel_dims
        )
        mask = mask.view(batch_shape + (1,) + spatial_shape).to(voxel_coordinates.dtype)
        mask = avg_pool_nd_function(n_spatial_dims)(mask, kernel_size=3, stride=1, padding=1) >= 1
        interpolated_mask = interpolate(
            volume=mask.to(voxel_coordinates.dtype),
            grid=voxel_coordinates,
            mode="bilinear",
            padding_mode="zeros",
        )
        return (
            interpolated_mask.view(
                get_batch_shape(interpolated_mask.shape, n_channel_dims=1)
                + channels_shape
                + get_spatial_shape(interpolated_mask.shape, n_channel_dims=1)
            )
            >= 1 - self._mask_threshold
        )

    @property
    def _kernel_size(self) -> Tuple[float, bool, bool]:
        return (4.0, False, False)

    def _evaluate_interpolation_kernel(self, coordinates: Tensor) -> Tensor:
        abs_coordinates = coordinates.abs()
        alpha = -0.75
        return ((alpha + 2) * abs_coordinates**3 - (alpha + 3) * abs_coordinates**2 + 1) * (
            abs_coordinates <= 1.0
        ) + (
            alpha * abs_coordinates**3
            - 5 * alpha * abs_coordinates**2
            + 8 * alpha * abs_coordinates
            - 4 * alpha
        ) * (
            (1 < abs_coordinates) & (abs_coordinates < 2)
        )


class _FlippingPermutation:
    def __init__(
        self, spatial_permutation: Sequence[int], flipped_spatial_dims: Sequence[int]
    ) -> None:
        self.spatial_permutation = spatial_permutation
        self.flipped_spatial_dims = flipped_spatial_dims

    @staticmethod
    def is_valid_permutation(n_dims: int, spatial_permutation: Sequence[int]) -> bool:
        """Check if the permutation is valid"""
        return set(spatial_permutation) == set(range(n_dims))

    def permute_sequence(self, sequence: Sequence) -> Tuple:
        """Permute a sequence"""
        if len(sequence) != len(self.spatial_permutation):
            raise ValueError("Sequence has wrong length")
        return tuple(sequence[spatial_dim] for spatial_dim in self.spatial_permutation)

    def __call__(self, volume: Tensor, n_channel_dims: int) -> Tensor:
        spatial_dims = get_spatial_dims(volume.ndim, n_channel_dims=n_channel_dims)
        if self.flipped_spatial_dims:
            flipped_dims = [spatial_dims[spatial_dim] for spatial_dim in self.flipped_spatial_dims]
            volume = volume.flip(dims=flipped_dims)
        volume = volume.permute(tuple(range(spatial_dims[0])) + self.permute_sequence(spatial_dims))
        return volume

    def _flipping_transformation(
        self,
        spatial_shape: Sequence[int],
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> HostDiagonalAffineTransformation:
        n_dims = len(spatial_shape)
        diagonal = ones(n_dims, dtype=dtype, device=torch_device("cpu"))
        translation = zeros(n_dims, dtype=dtype, device=torch_device("cpu"))
        for flipped_spatial_dim in self.flipped_spatial_dims:
            diagonal[flipped_spatial_dim] = -1.0
            translation[flipped_spatial_dim] = spatial_shape[flipped_spatial_dim] - 1
        return HostDiagonalAffineTransformation(
            diagonal=diagonal,
            translation=translation,
            device=device,
        )

    def as_transformation(
        self,
        spatial_shape: Sequence[int],
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> IHostAffineTransformation:
        """Return the transformation corresponding to the flipping and permutation"""
        if self.spatial_permutation == tuple(range(len(spatial_shape))):
            if not self.flipped_spatial_dims:
                return IdentityAffineTransformation(
                    n_dims=len(spatial_shape), dtype=dtype, device=device
                )
            return self._flipping_transformation(spatial_shape, dtype=dtype, device=device)
        matrix = self._flipping_transformation(
            spatial_shape, dtype=dtype, device=device
        ).as_matrix()
        matrix = matrix[tuple(self.spatial_permutation) + (-1,), :]
        return HostAffineTransformation(transformation_matrix_on_host=matrix, device=device)
