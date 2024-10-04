"""Interpolation class wrappers"""

from typing import Optional, Sequence

from torch import Tensor

from composable_mapping.mappable_tensor.mappable_tensor import (
    MappableTensor,
    PlainTensor,
)
from composable_mapping.util import combine_optional_masks, get_spatial_shape

from .dense_deformation import compute_fov_mask_at_voxel_coordinates, interpolate
from .interface import IInterpolator

MASK_THRESHOLD = 1e-5


class _BaseInterpolator(IInterpolator):
    """Linear interpolation in voxel coordinates"""

    def __init__(
        self,
        interpolation_mode: str,
        padding_mode: str = "border",
        mask_outside_fov: bool = True,
    ) -> None:
        self._interpolation_mode = interpolation_mode
        self._padding_mode = padding_mode
        self._mask_outside_fov = mask_outside_fov

    def interpolate_values(
        self, volume: Tensor, voxel_coordinates: Tensor, n_channel_dims: int = 1
    ) -> Tensor:
        # TODO: Handle n_channel_dims for interpolation
        return interpolate(
            volume,
            voxel_coordinates,
            mode=self._interpolation_mode,
            padding_mode=self._padding_mode,
        )

    def interpolate_mask(
        self,
        mask: Optional[Tensor],
        voxel_coordinates: Tensor,
        spatial_shape: Optional[Sequence[int]] = None,
        n_channel_dims: int = 1,
    ) -> Optional[Tensor]:
        # TODO: Handle n_channel_dims for interpolation
        interpolated_mask: Optional[Tensor] = None
        if mask is not None:
            interpolated_mask = interpolate(
                volume=mask.to(voxel_coordinates.dtype),
                grid=voxel_coordinates,
                mode="bilinear",
                padding_mode="zeros",
            )
            interpolated_mask = self._threshold_mask(interpolated_mask)
        fov_mask: Optional[Tensor] = None
        if (mask is None and self._mask_outside_fov) or (
            mask is not None and not self._mask_outside_fov
        ):
            if spatial_shape is None and mask is not None:
                spatial_shape = get_spatial_shape(mask.shape, n_channel_dims)
            elif spatial_shape is None:
                raise ValueError(
                    "Spatial shape must be provided for FOV masking when mask no mask is provided"
                )
            fov_mask = compute_fov_mask_at_voxel_coordinates(
                voxel_coordinates,
                volume_shape=spatial_shape,
            )
            if mask is not None and not self._mask_outside_fov:
                return interpolated_mask | (~fov_mask)
        return combine_optional_masks([fov_mask, interpolated_mask], n_channel_dims=n_channel_dims)

    def __call__(self, volume: MappableTensor, voxel_coordinates: MappableTensor) -> MappableTensor:
        if voxel_coordinates.n_channel_dims != 1:
            raise ValueError("Interpolation assumes single channel coordinates")
        coordinates_as_slice = voxel_coordinates.as_slice(volume.spatial_shape)
        data, data_mask = volume.generate(generate_missing_mask=False, cast_mask=False)
        coordinates_mask = voxel_coordinates.generate_mask(generate_missing_mask=False)
        if coordinates_as_slice is None:
            coordinate_values = voxel_coordinates.generate_values()
            values = self.interpolate_values(
                data, coordinate_values, n_channel_dims=volume.n_channel_dims
            )
            mask = self.interpolate_mask(
                data_mask,
                coordinate_values,
                spatial_shape=volume.spatial_shape,
                n_channel_dims=volume.n_channel_dims,
            )
        else:
            values = data[coordinates_as_slice]
            mask = data_mask[coordinates_as_slice] if data_mask is not None else None
        mask = combine_optional_masks(
            [mask, coordinates_mask], n_channel_dims=volume.n_channel_dims
        )
        return PlainTensor(values, mask, n_channel_dims=len(volume.channels_shape))

    def _threshold_mask(self, mask: Tensor) -> Tensor:
        return mask >= 1 - MASK_THRESHOLD


class LinearInterpolator(_BaseInterpolator):
    """Linear interpolation in voxel coordinates"""

    def __init__(
        self,
        padding_mode: str = "border",
    ) -> None:
        super().__init__("bilinear", padding_mode)


class NearestInterpolator(_BaseInterpolator):
    """Nearest neighbour interpolation in voxel coordinates"""

    def __init__(self, padding_mode: str = "border") -> None:
        super().__init__("nearest", padding_mode)


class BicubicInterpolator(_BaseInterpolator):
    """Bicubic interpolation in voxel coordinates"""

    def __init__(self, padding_mode: str = "border") -> None:
        super().__init__("bicubic", padding_mode)
