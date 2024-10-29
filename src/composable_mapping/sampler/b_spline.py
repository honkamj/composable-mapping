"""B-spline samplers"""

from typing import Tuple

from torch import Tensor

from .base import BaseSeparableSampler, SymmetricPolynomialKernelSupport
from .interface import LimitDirection


class CubicSplineSampler(BaseSeparableSampler):
    """Sampling based on regularly spaced cubic spline control points in voxel
    coordinates"""

    def __init__(
        self,
        prefilter: bool = False,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-4,
        mask_threshold: float = 1e-5,
    ) -> None:
        if prefilter:
            raise NotImplementedError(
                "Prefiltering is currently not implemented. "
                "Contact the developers if you would want it included."
            )
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            interpolating_sampler=prefilter,
            kernel_support=SymmetricPolynomialKernelSupport(
                kernel_width=lambda _: 4.0, polynomial_degree=lambda _: 3
            ),
            limit_direction=LimitDirection.RIGHT,
        )
        self._mask_threshold = mask_threshold

    def _kernel_parts(self, coordinates: Tensor) -> Tuple[Tensor, Tensor]:
        abs_coordinates = coordinates.abs()
        center_part = 2 / 3 + (0.5 * abs_coordinates - 1) * abs_coordinates**2
        surrounding_parts = -((abs_coordinates - 2) ** 3) / 6
        return center_part, surrounding_parts

    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        center_part, surrounding_parts = self._kernel_parts(coordinates)
        return (
            surrounding_parts * ((coordinates > -2) & (coordinates <= 1))
            + center_part * ((coordinates > -1) & (coordinates <= 0))
            + center_part * ((coordinates > 0) & (coordinates <= 1))
            + surrounding_parts * ((coordinates > 1) & (coordinates <= 2))
        )

    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        center_part, surrounding_parts = self._kernel_parts(coordinates)
        return (
            surrounding_parts * ((coordinates >= -2) & (coordinates < 1))
            + center_part * ((coordinates >= -1) & (coordinates < 0))
            + center_part * ((coordinates >= 0) & (coordinates < 1))
            + surrounding_parts * ((coordinates >= 1) & (coordinates < 2))
        )

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            "Sampling volume at arbitrary coordinates is not currently implemented. "
            "Contact the developers if you would want it included."
        )

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            "Sampling mask at arbitrary coordinates is not currently implemented. "
            "Contact the developers if you would want it included."
        )
