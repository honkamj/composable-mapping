"""Interpolation class wrappers"""

from torch import Tensor

from composable_mapping.mappable_tensor.mappable_tensor import (
    MappableTensor,
    PlainTensor,
)
from composable_mapping.util import (
    avg_pool_nd_function,
    combine_optional_masks,
    get_channels_shape,
    get_n_channel_dims,
    split_shape,
)

from .dense_deformation import interpolate
from .interface import IInterpolator


class _BaseInterpolator(IInterpolator):
    """Base interpolator in voxel coordinates"""

    def __init__(
        self,
        interpolation_mode: str,
        padding_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
    ) -> None:
        self._interpolation_mode = interpolation_mode
        self._padding_mode = padding_mode
        self._mask_extrapolated_regions_for_empty_volume_mask = (
            mask_extrapolated_regions_for_empty_volume_mask
        )

    def interpolate_values(
        self,
        volume: Tensor,
        voxel_coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            volume,
            voxel_coordinates,
            mode=self._interpolation_mode,
            padding_mode=self._padding_mode,
        )

    def __call__(self, volume: MappableTensor, voxel_coordinates: MappableTensor) -> MappableTensor:
        if voxel_coordinates.n_channel_dims != 1:
            raise ValueError("Interpolation assumes single channel coordinates")
        if voxel_coordinates.channels_shape[0] != len(volume.spatial_shape):
            raise ValueError("Interpolation assumes same number of channels as spatial dims")
        coordinates_as_slice = voxel_coordinates.as_slice(volume.spatial_shape)
        data = volume.generate_values()
        coordinates_mask = voxel_coordinates.generate_mask(
            generate_missing_mask=False, cast_mask=False
        )
        if coordinates_as_slice is None:
            coordinate_values = voxel_coordinates.generate_values()
            values = self.interpolate_values(
                data,
                coordinate_values,
            )
            mask = volume.generate_mask(
                generate_missing_mask=self._mask_extrapolated_regions_for_empty_volume_mask,
                cast_mask=False,
            )
            if mask is not None:
                mask = self.interpolate_mask(
                    mask,
                    coordinate_values,
                )
        else:
            values = data[coordinates_as_slice]
            mask = mask[coordinates_as_slice] if mask is not None else None
        mask = combine_optional_masks(
            mask, coordinates_mask, n_channel_dims=(volume.n_channel_dims, 1)
        )
        return PlainTensor(values, mask, n_channel_dims=len(volume.channels_shape))


class LinearInterpolator(_BaseInterpolator):
    """Linear interpolation in voxel coordinates"""

    def __init__(
        self,
        padding_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            interpolation_mode="bilinear",
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
        interpolated_mask = interpolate(
            volume=mask.to(voxel_coordinates.dtype),
            grid=voxel_coordinates,
            mode="bilinear",
            padding_mode="zeros",
        )
        return interpolated_mask >= 1 - self._mask_threshold


class NearestInterpolator(_BaseInterpolator):
    """Nearest neighbour interpolation in voxel coordinates"""

    def __init__(
        self,
        padding_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
    ) -> None:
        super().__init__(
            interpolation_mode="nearest",
            padding_mode=padding_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
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
