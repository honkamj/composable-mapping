"""Interpolation class wrappers"""

from abc import abstractmethod
from math import ceil, floor
from typing import Callable, List, Optional, Sequence, Tuple, Union

from numpy import argsort
from torch import Tensor
from torch import any as torch_any
from torch import arange
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import long, ones, ones_like, tensor, zeros
from torch.nn.functional import conv1d

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
    get_channels_shape,
    get_n_channel_dims,
    get_spatial_dims,
    includes_padding,
    is_croppable_first,
    split_shape,
)

from .dense_deformation import generate_voxel_coordinate_grid, interpolate
from .interface import IInterpolator


class _BaseSeparableInterpolator(IInterpolator):
    """Base interpolator in voxel coordinates which can be implemented as a
    separable convolution"""

    def __init__(
        self,
        interpolation_mode: str,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-5,
        mask_threshold: float = 1e-5,
    ) -> None:
        if extrapolation_mode not in ("zeros", "border", "reflection"):
            raise ValueError("Unknown extrapolation mode")
        if interpolation_mode not in ("nearest", "bilinear", "bicubic"):
            raise ValueError("Unknown interpolation mode")
        self._interpolation_mode = interpolation_mode
        self._extrapolation_mode = extrapolation_mode
        self._mask_extrapolated_regions_for_empty_volume_mask = (
            mask_extrapolated_regions_for_empty_volume_mask
        )
        self._convolution_threshold = convolution_threshold
        self._mask_threshold = mask_threshold

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

    def interpolate_values(
        self,
        values: Tensor,
        voxel_coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            values,
            voxel_coordinates,
            mode=self._interpolation_mode,
            padding_mode=self._extrapolation_mode,
        )

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

    def _interpolate_conv(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> Optional[MappableTensor]:
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
        translation = host_matrix[0, :-1, -1]
        exact_transformation = affine_transformation.cast(device=torch_device("cpu"))
        rounded_diagonal = diagonal.round()
        rounded_diagonal_transformation = DiagonalAffineTransformation(
            diagonal=rounded_diagonal, translation=translation
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
        rounded_translation = translation.round()
        rounded_diagonal_and_translation_transformation = DiagonalAffineTransformation(
            diagonal=rounded_diagonal, translation=rounded_translation
        )
        rounded_diagonal_and_translation_corner_points = (  # pylint: disable=not-callable
            rounded_diagonal_and_translation_transformation @ shape_transformation
        )(unit_grid)
        convolve_mask = (corner_points - rounded_diagonal_and_translation_corner_points).abs().amax(
            dim=get_batch_dims(unit_grid.ndim, n_channel_dims=1)
            + get_spatial_dims(unit_grid.ndim, n_channel_dims=1)
        ) > self._convolution_threshold
        floored_translation = translation.floor()
        relative_coordinate = translation - floored_translation
        rounded_integer_diagonal_list = rounded_diagonal.to(dtype=long).tolist()
        pads_or_crops: List[Tuple[int, int]] = []
        kernels: List[Tensor] = []
        (kernel_width, inclusive_min, inclusive_max) = self._kernel_size
        for (
            dim_size_volume,
            dim_size_grid,
            dim_convolve,
            dim_relative_coordinate,
            dim_floored_integer_translation,
            dim_integer_diagonal,
        ) in zip(
            flipping_permutation.permute_sequence(volume.spatial_shape),
            grid.spatial_shape,
            convolve_mask.tolist(),
            relative_coordinate.tolist(),
            floored_translation.to(dtype=long).tolist(),
            rounded_integer_diagonal_list,
        ):
            if not dim_convolve:
                dim_relative_coordinate = round(dim_relative_coordinate)
                shape_lower, shape_upper = (1, 0) if dim_relative_coordinate == 0.0 else (1, 0)
            else:
                shape_lower = self._get_flooring_function(inclusive_min)(
                    kernel_width / 2 - dim_relative_coordinate + 1
                )
                shape_upper = self._get_flooring_function(inclusive_max)(
                    kernel_width / 2 + dim_relative_coordinate
                )
            padding_lower = -dim_floored_integer_translation + shape_lower - 1
            padding_upper = (
                dim_floored_integer_translation
                + dim_integer_diagonal * (dim_size_grid - 1)
                - (dim_size_volume - 1)
                + shape_upper
            )
            pads_or_crops.append((padding_lower, padding_upper))
            kernels.append(
                self._evaluate_interpolation_kernel(
                    arange(shape_lower + shape_upper, device=grid.device, dtype=grid.dtype)
                    - (shape_lower - 1 + dim_relative_coordinate)
                )
            )
        padding_mode, padding_value = self._padding_mode_and_value
        if not is_croppable_first(
            spatial_shape=volume.spatial_shape, pads_or_crops=pads_or_crops, mode=padding_mode
        ):
            return None
        values, mask = volume.generate(
            generate_missing_mask=includes_padding(pads_or_crops)
            or self._mask_extrapolated_regions_for_empty_volume_mask,
            cast_mask=False,
        )
        values = flipping_permutation(values, n_channel_dims=volume.n_channel_dims)
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
            strides=rounded_integer_diagonal_list,
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
                    volume=(~interpolated_mask).to(dtype=grid.dtype),
                    kernels=[kernel.abs() for kernel in kernels],
                    strides=rounded_integer_diagonal_list,
                    n_channel_dims=volume.n_channel_dims,
                )
                <= self._mask_threshold
            )
        return PlainTensor(
            interpolated_values, interpolated_mask, n_channel_dims=volume.n_channel_dims
        )

    def _separable_conv_nd(
        self, volume: Tensor, kernels: Sequence[Tensor], strides: Sequence[int], n_channel_dims: int
    ) -> Tensor:
        if len(kernels) != len(strides) or len(kernels) != len(
            get_spatial_dims(volume.ndim, n_channel_dims)
        ):
            raise ValueError("Invalid number of kernels or strides")
        for spatial_dim, (kernel, stride) in enumerate(zip(kernels, strides)):
            if kernel.size(0) == 1:
                dim = get_spatial_dims(volume.ndim, n_channel_dims)[spatial_dim]
                volume = kernel * volume[(slice(None),) * dim + (slice(None, None, stride),)]
            else:
                volume = self._conv1d(
                    volume,
                    spatial_dim=spatial_dim,
                    kernel=kernel,
                    stride=stride,
                    n_channel_dims=n_channel_dims,
                )
        return volume

    @staticmethod
    def _conv1d(
        volume: Tensor, spatial_dim: int, kernel: Tensor, stride: int, n_channel_dims: int
    ) -> Tensor:
        dim = get_spatial_dims(volume.ndim, n_channel_dims)[spatial_dim]
        volume = volume.moveaxis(dim, -1)
        dim_excluded_shape = volume.shape[:-1]
        volume = volume.reshape(-1, 1, volume.size(-1))
        convolved = conv1d(  # pylint: disable=not-callable
            volume,
            kernel[None, None],
            bias=None,
            stride=(stride,),
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
        interpolated_values = self.interpolate_values(volume_values, coordinate_values)
        if volume_mask is not None:
            interpolated_mask: Optional[Tensor] = self.interpolate_mask(
                volume_mask,
                coordinate_values,
            )
        else:
            interpolated_mask = None
        return PlainTensor(
            interpolated_values, interpolated_mask, n_channel_dims=volume.n_channel_dims
        )


class LinearInterpolator(_BaseSeparableInterpolator):
    """Linear interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-5,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            interpolation_mode="bilinear",
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
        )

    @property
    def _kernel_size(self) -> Tuple[float, bool, bool]:
        return (2.0, False, False)

    def _evaluate_interpolation_kernel(self, coordinates: Tensor) -> Tensor:
        return 1 - coordinates.abs()

    def interpolate_mask(
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


class NearestInterpolator(_BaseSeparableInterpolator):
    """Nearest neighbour interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-5,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            interpolation_mode="nearest",
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
        )

    @property
    def _kernel_size(self) -> Tuple[float, bool, bool]:
        return (1.0, True, False)

    def _evaluate_interpolation_kernel(self, coordinates: Tensor) -> Tensor:
        return ones_like(coordinates)

    def interpolate_mask(
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


class BicubicInterpolator(_BaseSeparableInterpolator):
    """Bicubic interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-5,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            interpolation_mode="bicubic",
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
        )
        self._mask_threshold = mask_threshold

    @property
    def _kernel_size(self) -> Tuple[float, bool, bool]:
        return (4.0, False, False)

    def _evaluate_interpolation_kernel(self, coordinates: Tensor) -> Tensor:
        raise NotImplementedError("Not implemented!")

    def interpolate_mask(
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
        mask = avg_pool_nd_function(n_spatial_dims)(mask, kernel_size=3, stride=1, padding=1) > 0
        return (
            interpolate(
                volume=mask.to(voxel_coordinates.dtype),
                grid=voxel_coordinates,
                mode="bilinear",
                padding_mode="zeros",
            ).view(batch_shape + channels_shape + spatial_shape)
            >= 1 - self._mask_threshold
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

    def invert(self) -> "_FlippingPermutation":
        """Return the inverse transformation"""
        inverse_permutation = argsort(self.spatial_permutation)
        flipped_spatial_dims = [
            self.spatial_permutation[spatial_dim] for spatial_dim in self.flipped_spatial_dims
        ]
        return _FlippingPermutation(tuple(inverse_permutation), flipped_spatial_dims)
