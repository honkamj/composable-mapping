"""B-spline samplers"""

from typing import Tuple

from torch import Tensor

from composable_mapping.sampler.base import ISeparableKernelSupport

from .base import BaseSeparableSampler, NthDegreeSymmetricKernelSupport
from .interface import LimitDirection


class CubicSplineSampler(BaseSeparableSampler):
    """Sampling based on regularly spaced cubic spline control points in voxel
    coordinates

    Arguments:
        prefilter: Whether to prefilter the volume before sampling making
            the sampler an interpolator. Currently not implemented.
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
        prefilter: bool = False,
        extrapolation_mode: str = "border",
        mask_extrapolated_regions: bool = True,
        convolution_threshold: float = 1e-3,
        mask_threshold: float = 1e-5,
    ) -> None:
        if prefilter:
            raise NotImplementedError(
                "Prefiltering is currently not implemented. "
                "Contact the developers if you would want it included."
            )
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions=mask_extrapolated_regions,
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            limit_direction=LimitDirection.right(),
        )
        self._mask_threshold = mask_threshold
        self._prefilter = prefilter

    def _kernel_support(self, spatial_dim: int) -> ISeparableKernelSupport:
        return NthDegreeSymmetricKernelSupport(kernel_width=4.0, degree=3)

    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return self._prefilter

    def _piece_edges(self, spatial_dim: int) -> Tuple[int, ...]:
        return (-2, -1, 0, 1, 2)

    def _piecewise_kernel(self, coordinates: Tensor, spatial_dim: int, piece_index: int) -> Tensor:
        abs_coordinates = coordinates.abs()
        if piece_index in (0, 3):
            return -((abs_coordinates - 2) ** 3) / 6
        if piece_index in (1, 2):
            return 2 / 3 + (0.5 * abs_coordinates - 1) * abs_coordinates**2
        raise ValueError(f"Invalid piece index {piece_index}")

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
