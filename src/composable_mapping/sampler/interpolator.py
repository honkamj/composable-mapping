"""Interpolating samplers."""

from typing import Callable, Tuple

import torch.nn
from torch import Tensor, ones_like

from composable_mapping.sampler.base import ISeparableKernelSupport
from composable_mapping.util import (
    get_batch_shape,
    get_channels_shape,
    get_n_channel_dims,
    get_spatial_shape,
    split_shape,
)

from .base import BaseSeparableSampler, NthDegreeSymmetricKernelSupport
from .interface import LimitDirection
from .interpolate import interpolate


class LinearInterpolator(BaseSeparableSampler):
    """Linear interpolation in voxel coordinates

    Arguments:
        extrapolation_mode: Extrapolation mode for out-of-bound coordinates.
        mask_extrapolated_regions: Whether to mask extrapolated regions.
        convolution_threshold: Maximum allowed difference in coordinates
            for using convolution-based sampling (the difference might be upper
            bounded when doing the decision).
        mask_threshold: Maximum allowed weight for masked regions in a
            sampled location to still consider it valid (non-masked).
    """

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions: bool = True,
        convolution_threshold: float = 1e-3,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions=mask_extrapolated_regions,
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            limit_direction=LimitDirection.right(),
        )

    def _kernel_support(self, spatial_dim: int) -> ISeparableKernelSupport:
        return NthDegreeSymmetricKernelSupport(kernel_width=2.0, degree=1)

    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return True

    def _piece_edges(self, spatial_dim: int) -> Tuple[float, ...]:
        return (-1, 0, 1)

    def _piecewise_kernel(self, coordinates: Tensor, spatial_dim: int, piece_index: int) -> Tensor:
        if piece_index == 0:
            return 1 + coordinates
        if piece_index == 1:
            return 1 - coordinates
        raise ValueError(f"Invalid piece index {piece_index}")

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
        mask_extrapolated_regions: Whether to mask extrapolated regions.
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
        mask_extrapolated_regions: bool = True,
        convolution_threshold: float = 1e-3,
        mask_threshold: float = 1e-5,
        limit_direction: LimitDirection = LimitDirection.right(),
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions=mask_extrapolated_regions,
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            limit_direction=limit_direction,
        )

    def _kernel_support(self, spatial_dim: int) -> ISeparableKernelSupport:
        return NthDegreeSymmetricKernelSupport(kernel_width=1.0, degree=0)

    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return True

    def _piece_edges(self, spatial_dim: int) -> Tuple[float, ...]:
        return (-0.5, 0.5)

    def _piecewise_kernel(self, coordinates: Tensor, spatial_dim: int, piece_index: int) -> Tensor:
        return ones_like(coordinates)

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
        mask_extrapolated_regions: Whether to mask extrapolated regions.
        convolution_threshold: Maximum allowed difference in coordinates
            for using convolution-based sampling (the difference might be upper
            bounded when doing the decision).
        mask_threshold: Maximum allowed weight for masked regions in a
            sampled location to still consider it valid (non-masked).
    """

    def __init__(
        self,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions: bool = True,
        convolution_threshold: float = 1e-3,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions=mask_extrapolated_regions,
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            limit_direction=LimitDirection.right(),
        )
        self._mask_threshold = mask_threshold

    def _kernel_support(self, spatial_dim: int) -> ISeparableKernelSupport:
        return NthDegreeSymmetricKernelSupport(kernel_width=4.0, degree=3)

    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return True

    def _piece_edges(self, spatial_dim):
        return (-2, -1, 0, 1, 2)

    def _piecewise_kernel(self, coordinates: Tensor, spatial_dim: int, piece_index: int) -> Tensor:
        abs_coordinates = coordinates.abs()
        alpha = -0.75
        if piece_index in (0, 3):
            return alpha * (abs_coordinates**3 - 5 * abs_coordinates**2 + 8 * abs_coordinates - 4)
        if piece_index in (1, 2):
            return (alpha + 2) * abs_coordinates**3 - (alpha + 3) * abs_coordinates**2 + 1
        raise ValueError(f"Invalid piece index {piece_index}")

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
