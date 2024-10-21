"""Interpolation class wrappers"""

from abc import abstractmethod
from itertools import product
from math import ceil, floor
from typing import Callable, Optional, Sequence, Tuple

from numpy import argsort
from torch import Tensor, allclose
from torch import any as torch_any
from torch import arange
from torch import device as torch_device
from torch import floor_, long, ones, stack, tensor, zeros

from composable_mapping.mappable_tensor.affine_transformation import (
    DiagonalAffineTransformation,
    HostAffineTransformation,
    HostDiagonalAffineTransformation,
    IAffineTransformation,
    IdentityAffineTransformation,
    IHostAffineTransformation,
)
from composable_mapping.mappable_tensor.diagonal_matrix import (
    DiagonalAffineMatrixDefinition,
)
from composable_mapping.mappable_tensor.mappable_tensor import (
    MappableTensor,
    PlainTensor,
)
from composable_mapping.util import (
    avg_pool_nd_function,
    conv_nd_function,
    crop_and_then_pad_spatial,
    get_batch_dims,
    get_channels_shape,
    get_n_channel_dims,
    get_spatial_dims,
    get_spatial_shape,
    split_shape,
)

from .dense_deformation import generate_voxel_coordinate_grid, interpolate
from .interface import IInterpolator

SKIP_INTERPOLATION_THRESHOLD = 1e-5


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

    def __call__(self, volume: Tensor, n_channel_dims: int) -> Tensor:
        spatial_dims = get_spatial_dims(volume.ndim, n_channel_dims=n_channel_dims)
        if self.flipped_spatial_dims:
            flipped_dims = [spatial_dims[spatial_dim] for spatial_dim in self.flipped_spatial_dims]
            volume = volume.flip(dims=flipped_dims)
        volume = volume.permute(
            tuple(range(spatial_dims[0]))
            + tuple(spatial_dims[spatial_dim] for spatial_dim in self.spatial_permutation)
        )
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
        matrix = HostDiagonalAffineTransformation(
            diagonal=diagonal,
            translation=translation,
            device=device,
        ).as_matrix()
        return matrix

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


class _BaseInterpolator(IInterpolator):
    """Base interpolator in voxel coordinates"""

    def __init__(
        self,
        interpolation_mode: str,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-5,
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
    ) -> Optional[_FlippingPermutation]:
        largest_indices = matrix[:-1, :-1].argmax(dim=1).tolist()
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
        voxel_coordinates: MappableTensor,
        volume: MappableTensor,
    ) -> Optional[MappableTensor]:
        if voxel_coordinates.displacements is not None:
            return None
        grid = voxel_coordinates.grid
        affine_transformation = grid.affine_transformation
        assert grid is not None
        if affine_transformation.batch_shape not in ((1,), tuple()):
            return None
        channels_shape = affine_transformation.channels_shape
        if channels_shape[0] != channels_shape[1]:
            return None
        n_dims = len(grid.spatial_shape)
        host_matrix = voxel_coordinates.grid.affine_transformation.as_host_matrix()
        if host_matrix is None:
            return None
        host_matrix = host_matrix.view(n_dims + 1, n_dims + 1)
        flipping_permutation = self._obtain_flipping_permutation(host_matrix)
        if flipping_permutation is None:
            return None
        affine_transformation = (
            flipping_permutation.as_transformation(spatial_shape=grid.spatial_shape)
            @ affine_transformation
        )
        host_matrix = affine_transformation.as_host_matrix()
        assert host_matrix is not None
        diagonal = host_matrix[:-1, :-1].diagonal()
        translation = host_matrix[:-1, -1]
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
        rounded_diagonal_corner_points = (rounded_diagonal_transformation @ shape_transformation)(
            unit_grid
        )
        max_rounded_diagonal_displacements = (
            (corner_points - rounded_diagonal_corner_points)
            .abs()
            .amax(dim=get_spatial_dims(unit_grid.ndim, n_channel_dims=1))
        )
        if torch_any(max_rounded_diagonal_displacements > self._convolution_threshold):
            return None
        rounded_translation = translation.round()
        rounded_diagonal_and_translation_transformation = DiagonalAffineTransformation(
            diagonal=rounded_diagonal, translation=rounded_translation
        )
        rounded_diagonal_and_translation_corner_points = (
            rounded_diagonal_and_translation_transformation @ shape_transformation
        )(unit_grid)
        max_rounded_diagonal_and_translation_displacements = (
            (corner_points - rounded_diagonal_and_translation_corner_points)
            .abs()
            .amax(dim=get_spatial_dims(unit_grid.ndim, n_channel_dims=1))
        )
        convolve_mask = (
            max_rounded_diagonal_and_translation_displacements.abs().amax(
                dim=get_spatial_dims(unit_grid.ndim, n_channel_dims=1)
                + get_batch_dims(unit_grid.ndim, n_channel_dims=1)
            )
            <= self._convolution_threshold
        )
        floored_translation = translation.floor()
        relative_coordinate = translation - floored_translation
        kernel_shape = []
        paddings = []
        kernel_coordinates = []
        for (
            (kernel_width, inclusive_min, inclusive_max),
            dim_size,
            dim_convolve,
            dim_relative_coordinate,
            dim_floored_integer_translation,
            dim_integer_diagonal,
        ) in zip(
            self._kernel_size,
            grid.spatial_shape,
            convolve_mask.tolist(),
            relative_coordinate.tolist(),
            floored_translation.to(dtype=long).tolist(),
            rounded_diagonal.to(dtype=long).tolist(),
        ):
            if not dim_convolve:
                shape_lower, shape_upper = 1, 0 if dim_relative_coordinate <= 0.5 else 1, 0
            else:
                shape_lower = self._get_flooring_function(inclusive_min)(
                    kernel_width / 2 - dim_relative_coordinate
                )
                shape_upper = self._get_flooring_function(inclusive_max)(
                    kernel_width / 2 + dim_relative_coordinate - 1
                )
                kernel_dim_size = shape_lower + shape_upper
            kernel_shape.append(kernel_dim_size)
            padding_lower = -dim_floored_integer_translation + shape_lower - 1
            padding_upper = (
                dim_floored_integer_translation
                + (dim_integer_diagonal - 1) * (dim_size - 1)
                + shape_upper
            )
            paddings.append((padding_lower, padding_upper))
            kernel_coordinates.append(
                arange(shape_lower + shape_upper, device=torch_device("cpu"), dtype=grid.dtype)
                - shape_lower
                - 1
                + dim_relative_coordinate
            )
        # kernel_size / 2 - coord

    @staticmethod
    def _get_flooring_function(inclusive: bool) -> Callable[[float], float]:
        if inclusive:
            return lambda x: int(floor(x))
        return lambda x: int(ceil(x - 1))

    @abstractmethod
    def _kernel_size(self) -> Sequence[Tuple[float, bool, bool]]:
        """Return the kernel size for the convolution and whether min and max
        are inclusive or not"""

    def _as_integer_host_diagonal_and_translation(
        self, voxel_coordinates: MappableTensor
    ) -> Optional[Tuple[Tensor, Tensor]]:
        if voxel_coordinates.displacements is not None:
            return None
        assert voxel_coordinates.grid is not None
        if voxel_coordinates.grid.affine_transformation.batch_shape not in ((1,), tuple()):
            return None
        channels_shape = voxel_coordinates.grid.affine_transformation.channels_shape
        if channels_shape[0] != channels_shape[1]:
            return None
        n_dims = channels_shape[0] - 1
        host_diagonal_matrix = voxel_coordinates.grid.affine_transformation.as_host_diagonal()
        if host_diagonal_matrix is not None:
            diagonal = host_diagonal_matrix.generate_diagonal().view(n_dims)
            translation = host_diagonal_matrix.generate_translation().view(n_dims)
        else:
            host_matrix = voxel_coordinates.grid.affine_transformation.as_host_matrix()
            if host_matrix is None:
                return None
            if not voxel_coordinates.grid.affine_transformation.is_diagonal():
                return None
            host_matrix = host_matrix.view(n_dims + 1, n_dims + 1)
            diagonal = host_matrix[:-1, :-1].diagonal()
            translation = host_matrix[:-1, -1]
        if not allclose(diagonal.round(), diagonal, atol=SKIP_INTERPOLATION_THRESHOLD):
            return None
        return (
            diagonal.to(dtype=long),
            translation,
        )

    def __call__(self, volume: MappableTensor, voxel_coordinates: MappableTensor) -> MappableTensor:
        if voxel_coordinates.n_channel_dims != 1:
            raise ValueError("Interpolation assumes single channel coordinates")
        if voxel_coordinates.channels_shape[0] != len(volume.spatial_shape):
            raise ValueError("Interpolation assumes same number of channels as spatial dims")
        integer_host_diagonal_and_translation = self._as_integer_host_diagonal_and_translation(
            voxel_coordinates
        )
        if integer_host_diagonal_and_translation is None:
            return self._interpolate_grid_sample(volume, voxel_coordinates)
        integer_host_diagonal, host_translation = integer_host_diagonal_and_translation
        negative_diagonal_mask = integer_host_diagonal < 0
        host_translation[negative_diagonal_mask] = (
            -host_translation[negative_diagonal_mask]
            + host_translation.new(voxel_coordinates.spatial_shape)
            - 1
        )
        host_conv_kernel_and_pads_or_crops = self._as_host_conv_kernel_and_pads_or_crops(
            integer_host_diagonal.abs(),
            host_translation,
            voxel_coordinates.spatial_shape,
            volume.spatial_shape,
        )
        if host_conv_kernel_and_pads_or_crops is None:
            return self._interpolate_grid_sample(volume, voxel_coordinates)
        cpu_conv_kernel, pads_or_crops = host_conv_kernel_and_pads_or_crops
        padding_mode, _value = self._padding_mode_and_value
        if not is_croppable_first(pads_or_crops, mode=padding_mode):
            return self._interpolate_grid_sample(volume, voxel_coordinates)
        flipped_spatial_dims = negative_diagonal_mask.nonzero().view(-1)
        return self._interpolate_conv(
            volume, voxel_coordinates, cpu_conv_kernel, pads_or_crops, flipped_spatial_dims.tolist()
        )

    @abstractmethod
    def _interpolate_conv(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
        host_conv_kernel: Tensor,
        pads_or_crops: Sequence[Tuple[int, int]],
        flipped_spatial_dims: Sequence[int],
    ) -> MappableTensor:
        padding_mode, value = self._padding_mode_and_value
        values, mask = volume.generate(
            generate_missing_mask=includes_padding(pads_or_crops),
            cast_mask=False,
        )
        padded_values = crop_and_then_pad_spatial(
            values,
            pads_or_crops,
            mode=padding_mode,
            value=value,
            n_channel_dims=volume.n_channel_dims,
        )
        flipped_dimensions = [
            dim
            for spatial_dim, dim in enumerate(
                get_spatial_dims(len(volume.shape), volume.n_channel_dims)
            )
            if spatial_dim in flipped_spatial_dims
        ]
        if flipped_dimensions:
            volume = volume.modify_values(volume.generate_values().flip(dims=flipped_dimensions))
        if mask is None:
            padded_mask: Optional[Tensor] = None
        else:
            padded_mask = crop_and_then_pad_spatial(
                mask,
                pads_or_crops,
                mode="constant",
                value=False,
                n_channel_dims=volume.n_channel_dims,
            )
        if host_conv_kernel.shape.numel() == 1:
            if host_conv_kernel.view(-1)[0].item() != 1:
                padded_values = padded_values * host_conv_kernel.view(-1)[0].item()
            return PlainTensor(padded_values, padded_mask, n_channel_dims=volume.n_channel_dims)
        padded_spatial_shape = get_spatial_shape(
            padded_values.shape, n_channel_dims=volume.n_channel_dims
        )
        conv_kernel = host_conv_kernel.to(
            padded_values.device, non_blocking=padded_values.device.type != "cpu"
        )
        interpolated_values = conv_nd_function(len(padded_spatial_shape))(
            padded_values.view(-1, 1, *padded_spatial_shape),
            conv_kernel[None, None],
        )
        interpolated_values = interpolated_values.view(
            volume.batch_shape
            + volume.channels_shape
            + get_spatial_shape(interpolated_values.shape, n_channel_dims=1)
        )
        if padded_mask is None:
            interpolated_mask: Optional[Tensor] = None
        else:
            interpolated_mask = conv_nd_function(len(padded_spatial_shape))(
                padded_mask.view(-1, 1, *padded_spatial_shape).to(padded_values.dtype),
                conv_kernel[None, None],
            )
            interpolated_mask = (
                interpolated_mask.view(
                    volume.batch_shape
                    + get_channels_shape(volume.mask_shape, n_channel_dims=volume.n_channel_dims)
                    + get_spatial_shape(interpolated_mask.shape, n_channel_dims=1)
                )
                >= 1 - SKIP_INTERPOLATION_THRESHOLD
            )
        return PlainTensor(
            interpolated_values, interpolated_mask, n_channel_dims=volume.n_channel_dims
        )

    @abstractmethod
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

    @abstractmethod
    def _as_host_conv_kernel_and_pads_or_crops(
        self,
        positive_integer_host_diagonal: Tensor,
        host_translation: Tensor,
        interpolation_grid_spatial_shape: Tuple[int, ...],
        volume_spatial_shape: Tuple[int, ...],
    ) -> Tuple[Tensor, Sequence[Tuple[int, int]]]:
        """Return the voxel coordinates as a CPU convolution kernel and paddings
        if the interpolation can be represented as a convolution"""


class LinearInterpolator(_BaseInterpolator):
    """Linear interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
    ) -> None:
        super().__init__(
            interpolation_mode="bilinear",
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
        )

    def _as_host_conv_kernel_and_pads_or_crops(
        self,
        positive_integer_host_diagonal: Tensor,
        host_translation: Tensor,
        interpolation_grid_spatial_shape: Tuple[int, ...],
        volume_spatial_shape: Tuple[int, ...],
    ) -> Tuple[Tensor, Sequence[Tuple[int, int]]]:
        rounded_translation = host_translation.round()
        grid_shape_tensor = positive_integer_host_diagonal.new_tensor(
            interpolation_grid_spatial_shape
        )
        volume_shape_tensor = positive_integer_host_diagonal.new_tensor(volume_spatial_shape)
        if allclose(rounded_translation, host_translation, atol=SKIP_INTERPOLATION_THRESHOLD):
            integer_translation = rounded_translation.to(dtype=positive_integer_host_diagonal.dtype)
            padding_lower = -integer_translation
            padding_upper = (
                integer_translation
                + positive_integer_host_diagonal * grid_shape_tensor
                - volume_shape_tensor
                + 1
            )
            return host_translation.new_ones(1), tuple(
                zip(padding_lower.tolist(), padding_upper.tolist())
            )
        ceil_translation = (host_translation + 1).floor()
        floor_translation = host_translation.floor()
        floor_and_ceil_translation = stack((floor_translation, ceil_translation), dim=-1)
        n_dims = len(interpolation_grid_spatial_shape)
        kernel = host_translation.new_empty((2,) * n_dims)
        for corner_index in product(range(2), repeat=n_dims):
            corner = floor_and_ceil_translation[tuple(range(n_dims)), corner_index]
            kernel[corner_index] = (1 - (host_translation - corner).abs()).prod()
        padding_lower = -floor_translation.to(dtype=positive_integer_host_diagonal.dtype)
        padding_upper = (
            ceil_translation.to(dtype=positive_integer_host_diagonal.dtype)
            + positive_integer_host_diagonal * grid_shape_tensor
            - volume_shape_tensor
            + 1
        )
        return kernel, tuple(zip(padding_lower.tolist(), padding_upper.tolist()))

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
        return interpolated_mask >= 1 - SKIP_INTERPOLATION_THRESHOLD


class NearestInterpolator(_BaseInterpolator):
    """Nearest neighbour interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
    ) -> None:
        super().__init__(
            interpolation_mode="nearest",
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
        )

    def _as_host_conv_kernel_and_pads_or_crops(
        self,
        positive_integer_host_diagonal: Tensor,
        host_translation: Tensor,
        interpolation_grid_spatial_shape: Tuple[int, ...],
        volume_spatial_shape: Tuple[int, ...],
    ) -> Tuple[Tensor, Sequence[Tuple[int, int]]]:
        integer_translation = host_translation.round().to(
            dtype=positive_integer_host_diagonal.dtype
        )
        padding_lower = -integer_translation
        padding_upper = (
            integer_translation
            + positive_integer_host_diagonal
            * positive_integer_host_diagonal.new_tensor(interpolation_grid_spatial_shape)
            - positive_integer_host_diagonal.new_tensor(volume_spatial_shape)
            + 1
        )
        return host_translation.new_ones(1), tuple(
            zip(padding_lower.tolist(), padding_upper.tolist())
        )

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


class BicubicInterpolator(_BaseInterpolator):
    """Bicubic interpolation in voxel coordinates"""

    def __init__(
        self,
        padding_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            interpolation_mode="bicubic",
            padding_mode=padding_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
        )
        self._mask_threshold = mask_threshold

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
        mask = avg_pool_nd_function(n_spatial_dims)(mask, kernel_size=3, stride=1, padding=1) >= 1
        return (
            interpolate(
                volume=mask.to(voxel_coordinates.dtype),
                grid=voxel_coordinates,
                mode="bilinear",
                padding_mode="zeros",
            ).view(batch_shape + channels_shape + spatial_shape)
            >= 1 - self._mask_threshold
        )
