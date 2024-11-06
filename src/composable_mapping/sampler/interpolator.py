"""Interpolating samplers."""

from typing import Callable, Tuple

import torch.nn
from torch import Tensor, ones_like

from composable_mapping.dense_deformation import interpolate
from composable_mapping.sampler.base import ISeparableKernelSupport
from composable_mapping.util import (
    get_batch_shape,
    get_channels_shape,
    get_n_channel_dims,
    get_spatial_shape,
    split_shape,
)

from .base import BaseSeparableSampler, SymmetricPolynomialKernelSupport
from .interface import LimitDirection


class LinearInterpolator(BaseSeparableSampler):
    """Linear interpolation in voxel coordinates

    Arguments:
        extrapolation_mode: Extrapolation mode for out-of-bound coordinates.
        mask_extrapolated_regions_for_empty_volume_mask: Whether to mask
            extrapolated regions when input volume mask is empty.
        convolution_threshold: Maximum allowed difference in coordinates
            for using convolution-based sampling (the difference might be upper
            bounded when doing the decision).
        mask_threshold: Maximum allowed weight for masked regions in a
            sampled location to still consider it valid (non-masked).
    """

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
            limit_direction=LimitDirection.right(),
        )

    def _kernel_support(self, spatial_dim: int) -> ISeparableKernelSupport:
        return SymmetricPolynomialKernelSupport(kernel_width=2.0, polynomial_degree=1)

    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
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
    """Nearest neighbour interpolation in voxel coordinates.

    Arguments:
        extrapolation_mode: Extrapolation mode for out-of-bound coordinates.
        mask_extrapolated_regions_for_empty_volume_mask: Whether to mask
            extrapolated regions when input volume mask is empty.
        convolution_threshold: Maximum allowed difference in coordinates
            for using convolution-based sampling (the difference might be upper
            bounded when doing the decision).
        mask_threshold: Maximum allowed weight for masked regions in a
            sampled location to still consider it valid (non-masked).
        limit_direction: How to handle points at equal distances. This
            option currently applies only to convolution based sampling.
    """

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-4,
        mask_threshold: float = 1e-5,
        limit_direction: LimitDirection = LimitDirection.right(),
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            limit_direction=limit_direction,
        )

    def _kernel_support(self, spatial_dim: int) -> ISeparableKernelSupport:
        return SymmetricPolynomialKernelSupport(kernel_width=1.0, polynomial_degree=0)

    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
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
    """Bicubic interpolation in voxel coordinates.

    Arguments:
        extrapolation_mode: Extrapolation mode for out-of-bound coordinates.
        mask_extrapolated_regions_for_empty_volume_mask: Whether to mask
            extrapolated regions when input volume mask is empty.
        convolution_threshold: Maximum allowed difference in coordinates
            for using convolution-based sampling (the difference might be upper
            bounded when doing the decision).
        mask_threshold: Maximum allowed weight for masked regions in a
            sampled location to still consider it valid (non-masked).
    """

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
            limit_direction=LimitDirection.right(),
        )
        self._mask_threshold = mask_threshold

    def _kernel_support(self, spatial_dim: int) -> ISeparableKernelSupport:
        return SymmetricPolynomialKernelSupport(kernel_width=4.0, polynomial_degree=3)

    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return True

    def _kernel_parts(self, coordinates: Tensor) -> Tuple[Tensor, Tensor]:
        abs_coordinates = coordinates.abs()
        alpha = -0.75
        center_part = (alpha + 2) * abs_coordinates**3 - (alpha + 3) * abs_coordinates**2 + 1
        surrounding_part = alpha * (
            abs_coordinates**3 - 5 * abs_coordinates**2 + 8 * abs_coordinates - 4
        )
        return center_part, surrounding_part

    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        center_part, surrounding_part = self._kernel_parts(coordinates)
        return (
            surrounding_part * ((coordinates > -2) & (coordinates <= -1))
            + center_part * ((coordinates > -1) & (coordinates <= 0))
            + center_part * ((coordinates > 0) & (coordinates <= 1))
            + surrounding_part * ((coordinates > 1) & (coordinates <= 2))
        )

    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        center_part, surrounding_part = self._kernel_parts(coordinates)
        return (
            surrounding_part * ((coordinates >= -2) & (coordinates < -1))
            + center_part * ((coordinates >= -1) & (coordinates < 0))
            + center_part * ((coordinates >= 0) & (coordinates < 1))
            + surrounding_part * ((coordinates >= 1) & (coordinates < 2))
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
        mask = _avg_pool_nd_function(n_spatial_dims)(mask, kernel_size=3, stride=1, padding=1) >= 1
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


def _avg_pool_nd_function(n_dims: int) -> Callable[..., Tensor]:
    return getattr(torch.nn.functional, f"avg_pool{n_dims}d")
