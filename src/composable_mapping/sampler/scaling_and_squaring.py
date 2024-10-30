"""Scaling and squaring sampler"""

from typing import TYPE_CHECKING, Any, Mapping, Optional

from torch import Tensor

from composable_mapping.mappable_tensor import MappableTensor, voxel_grid
from composable_mapping.util import get_spatial_shape

from .interface import DataFormat, ISampler, LimitDirection

if TYPE_CHECKING:
    from composable_mapping.coordinate_system import CoordinateSystem


class ScalingAndSquaring(ISampler):
    """Scaling and squaring sampler"""

    def __init__(
        self,
        sampler: ISampler,
        steps: int,
        inverse: bool = False,
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
    ) -> None:
        self._sampler = sampler
        self._steps = steps
        self._inverse = inverse
        self._mask_extrapolated_regions_for_empty_volume_mask = (
            mask_extrapolated_regions_for_empty_volume_mask
        )

    def __call__(self, volume: MappableTensor, coordinates: MappableTensor) -> MappableTensor:
        if volume.n_channel_dims != 1 or volume.channels_shape[0] != len(volume.spatial_shape):
            raise ValueError(
                "Scaling and squaring sampler assumes single channel displacements "
                "with same number of channels as spatial dims."
            )
        ddf = self._integrate_svf(volume.generate_values())
        return self._sampler(volume.modify_values(ddf), coordinates)

    def derivative(
        self, spatial_dim: int, limit_direction: LimitDirection = LimitDirection.AVERAGE
    ) -> "ISampler":
        raise NotImplementedError("Derivative sampling is not implemented for scaling and squaring")

    def inverse(
        self,
        coordinate_system: "CoordinateSystem",
        data_format: DataFormat,
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> ISampler:
        if data_format.coordinate_type == "voxel" and data_format.representation == "displacements":
            return ScalingAndSquaring(self._sampler, self._steps, inverse=not self._inverse)
        raise ValueError(
            "The sampler has been currently implemented only for voxel displacements data format."
        )

    def _integrate_svf(self, svf: Tensor) -> Tensor:
        if self._inverse:
            svf = -svf
        spatial_shape = get_spatial_shape(svf.shape, n_channel_dims=1)
        grid = voxel_grid(spatial_shape, dtype=svf.dtype, device=svf.device).generate_values()
        integrated = svf / 2**self._steps
        for _ in range(self._steps):
            integrated = self._sampler.sample_values(integrated, integrated + grid) + integrated
        return integrated

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return self._sampler.sample_values(self._integrate_svf(volume), coordinates)

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return self._sampler.sample_mask(mask, coordinates)