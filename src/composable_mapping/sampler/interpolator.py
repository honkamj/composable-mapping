"""Interpolation class wrappers"""

from typing import Tuple

from torch import Tensor, ones_like

from composable_mapping.dense_deformation import interpolate
from composable_mapping.util import (
    avg_pool_nd_function,
    get_batch_shape,
    get_channels_shape,
    get_n_channel_dims,
    get_spatial_shape,
    split_shape,
)

from .base import BaseSeparableSampler, SymmetricPolynomialKernelSupport
from .interface import LimitDirection


class LinearInterpolator(BaseSeparableSampler):
    """Linear interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-4,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            kernel_support=SymmetricPolynomialKernelSupport(
                kernel_width=lambda _: 2.0, polynomial_degree=lambda _: 1
            ),
            limit_direction=LimitDirection.RIGHT,
        )

    def _interpolating_kernel(self, spatial_dim: int) -> bool:
        return True

    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        return (1 + coordinates) * ((coordinates > -1) & (coordinates <= 0)) + (1 - coordinates) * (
            (coordinates > 0) & (coordinates <= 1)
        )

    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        return (1 + coordinates) * ((coordinates >= -1) & (coordinates < 0)) + (1 - coordinates) * (
            (coordinates >= 0) & (coordinates < 1)
        )

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            volume,
            coordinates,
            mode="bilinear",
            padding_mode=self._extrapolation_mode,
        )

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        interpolated_mask = interpolate(
            volume=mask.to(coordinates.dtype),
            grid=coordinates,
            mode="bilinear",
            padding_mode="zeros",
        )
        return interpolated_mask >= 1 - self._mask_threshold


class NearestInterpolator(BaseSeparableSampler):
    """Nearest neighbour interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-4,
        mask_threshold: float = 1e-5,
        limit_direction: LimitDirection = LimitDirection.RIGHT,
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            kernel_support=SymmetricPolynomialKernelSupport(
                kernel_width=lambda _: 1.0, polynomial_degree=lambda _: 0
            ),
            limit_direction=limit_direction,
        )

    def _interpolating_kernel(self, spatial_dim: int) -> bool:
        return True

    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        return ones_like(coordinates) * ((coordinates > -0.5) & (coordinates <= 0.5))

    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        return ones_like(coordinates) * ((coordinates >= -0.5) & (coordinates < 0.5))

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            volume,
            coordinates,
            mode="nearest",
            padding_mode=self._extrapolation_mode,
        )

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            volume=mask.to(coordinates.dtype),
            grid=coordinates,
            mode="nearest",
            padding_mode="zeros",
        ).to(mask.dtype)


class BicubicInterpolator(BaseSeparableSampler):
    """Bicubic interpolation in voxel coordinates"""

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-4,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            kernel_support=SymmetricPolynomialKernelSupport(
                kernel_width=lambda _: 4.0, polynomial_degree=lambda _: 3
            ),
            limit_direction=LimitDirection.RIGHT,
        )
        self._mask_threshold = mask_threshold

    def _interpolating_kernel(self, spatial_dim: int) -> bool:
        return True

    def _kernel_parts(self, coordinates: Tensor) -> Tuple[Tensor, Tensor]:
        abs_coordinates = coordinates.abs()
        alpha = -0.75
        center_part = (alpha + 2) * abs_coordinates**3 - (alpha + 3) * abs_coordinates**2 + 1
        surrounding_parts = alpha * (
            abs_coordinates**3 - 5 * abs_coordinates**2 + 8 * abs_coordinates - 4
        )
        return center_part, surrounding_parts

    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        center_part, surrounding_parts = self._kernel_parts(coordinates)
        return (
            surrounding_parts * ((coordinates > -2) & (coordinates <= -1))
            + center_part * ((coordinates > -1) & (coordinates <= 0))
            + center_part * ((coordinates > 0) & (coordinates <= 1))
            + surrounding_parts * ((coordinates > 1) & (coordinates <= 2))
        )

    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        center_part, surrounding_parts = self._kernel_parts(coordinates)
        return (
            surrounding_parts * ((coordinates >= -2) & (coordinates < -1))
            + center_part * ((coordinates >= -1) & (coordinates < 0))
            + center_part * ((coordinates >= 0) & (coordinates < 1))
            + surrounding_parts * ((coordinates >= 1) & (coordinates < 2))
        )

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return interpolate(
            volume,
            coordinates,
            mode="bicubic",
            padding_mode=self._extrapolation_mode,
        )

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        n_spatial_dims = get_channels_shape(coordinates.shape, n_channel_dims=1)[0]
        n_channel_dims = get_n_channel_dims(mask.ndim, n_spatial_dims)
        batch_shape, channels_shape, spatial_shape = split_shape(
            mask.shape, n_channel_dims=n_channel_dims
        )
        mask = mask.view(batch_shape + (1,) + spatial_shape).to(coordinates.dtype)
        mask = avg_pool_nd_function(n_spatial_dims)(mask, kernel_size=3, stride=1, padding=1) >= 1
        interpolated_mask = interpolate(
            volume=mask.to(coordinates.dtype),
            grid=coordinates,
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
