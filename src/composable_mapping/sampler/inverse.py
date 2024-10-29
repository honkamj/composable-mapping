"""Displacement inversion sampler"""

from typing import TYPE_CHECKING, Any, Mapping, Optional

from deformation_inversion_layer import (
    DeformationInversionArguments,
    fixed_point_invert_deformation,
)
from deformation_inversion_layer.interface import FixedPointSolver
from torch import Tensor
from torch import dtype as torch_dtype

from composable_mapping.mappable_tensor import MappableTensor, mappable
from composable_mapping.util import combine_optional_masks

from .interface import DataFormat, ISampler, LimitDirection

if TYPE_CHECKING:
    from composable_mapping.coordinate_system import CoordinateSystem


class FixedPointInverseSampler(ISampler):
    """Displacement field inversion sampler

    Assumes that the sampled data is a displacement field in voxel coordinates.
    """

    def __init__(
        self,
        sampler: ISampler,
        forward_solver: Optional[FixedPointSolver] = None,
        backward_solver: Optional[FixedPointSolver] = None,
        forward_dtype: Optional[torch_dtype] = None,
        backward_dtype: Optional[torch_dtype] = None,
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
    ):
        self._sampler = sampler
        self._mask_extrapolated_regions_for_empty_volume_mask = (
            mask_extrapolated_regions_for_empty_volume_mask
        )
        self._inversion_arguments = DeformationInversionArguments(
            interpolator=sampler.sample_values,
            forward_solver=forward_solver,
            backward_solver=backward_solver,
            forward_dtype=forward_dtype,
            backward_dtype=backward_dtype,
        )

    def __call__(self, volume: MappableTensor, coordinates: MappableTensor) -> MappableTensor:
        if coordinates.n_channel_dims != 1:
            raise ValueError("Interpolation assumes single channel coordinates")
        if coordinates.channels_shape[0] != len(volume.spatial_shape):
            raise ValueError("Interpolation assumes same number of channels as spatial dims")
        if volume.n_channel_dims != 1 or volume.channels_shape[0] != len(volume.spatial_shape):
            raise ValueError(
                "Displacement field inverse sampler assumes single channel displacement "
                "with same number of channels as spatial dims."
            )
        data, data_mask = volume.generate(
            generate_missing_mask=self._mask_extrapolated_regions_for_empty_volume_mask,
            cast_mask=False,
        )
        coordinates_values, coordinates_mask = coordinates.generate(
            generate_missing_mask=False, cast_mask=False
        )
        inverted_values = self._sample_values(
            data, coordinates_values, -self._sampler(volume, coordinates).generate_values()
        )
        if data_mask is None:
            mask: Optional[Tensor] = None
        else:
            mask = self._sampler.sample_mask(
                mask=data_mask,
                coordinates=coordinates_values + inverted_values,
            )
        mask = combine_optional_masks(
            mask,
            coordinates_mask,
            n_channel_dims=(volume.n_channel_dims, coordinates.n_channel_dims),
        )
        return mappable(inverted_values, mask, n_channel_dims=volume.n_channel_dims)

    def derivative(
        self, spatial_dim: int, limit_direction: LimitDirection = LimitDirection.AVERAGE
    ) -> "ISampler":
        raise NotImplementedError(
            "Derivative sampling is not implemented for the inverse sampler. "
        )

    def inverse(
        self,
        coordinate_system: "CoordinateSystem",
        data_format: DataFormat,
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> ISampler:
        if data_format.coordinate_type == "voxel" and data_format.representation == "displacements":
            return self._sampler
        raise ValueError(
            "The sampler has been currently implemented only for voxel displacements data format."
        )

    def _sample_values(
        self, volume: Tensor, coordinates: Tensor, initial_guess: Optional[Tensor]
    ) -> Tensor:
        return fixed_point_invert_deformation(
            displacement_field=volume,
            arguments=self._inversion_arguments,
            initial_guess=initial_guess,
            coordinates=coordinates,
        )

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        return self._sample_values(volume, coordinates, None)

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise RuntimeError("Mask can not be sampled without the displacement field")
