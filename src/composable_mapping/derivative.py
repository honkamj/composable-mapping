"""Estimate spatial derivatives of composable mappings"""

from typing import Optional

from composable_mapping.composable_mapping import (
    GridComposableMapping,
    ICoordinateSystemContainer,
)
from composable_mapping.mappable_tensor import MappableTensor, stack_mappable_tensors
from composable_mapping.sampler import ISampler, LimitDirection, get_sampler


def estimate_spatial_derivatives(
    mapping: GridComposableMapping,
    spatial_dim: int,
    target: Optional[ICoordinateSystemContainer] = None,
    limit_direction: LimitDirection = LimitDirection.AVERAGE,
    sampler: Optional[ISampler] = None,
) -> MappableTensor:
    """Estimate spatial derivatives of a grid composable mapping

    Derivatives are computed with respect to space rotated according to the
    target coordinate system.
    """
    if target is None:
        target = mapping.coordinate_system
    if sampler is None:
        sampler = get_sampler(sampler)
    return (
        mapping.resample_to(
            mapping,
            sampler=sampler.derivative(spatial_dim=spatial_dim, limit_direction=limit_direction),
        ).sample_to(target)
        / target.coordinate_system.grid_spacing()[spatial_dim]
    )


def estimate_spatial_jacobian_matrices(
    mapping: GridComposableMapping,
    target: Optional[ICoordinateSystemContainer] = None,
    limit_direction: LimitDirection = LimitDirection.AVERAGE,
    sampler: Optional[ISampler] = None,
) -> MappableTensor:
    """Estimate spatial Jacobian matrices of a grid composable mapping

    Jacobian matrices are computed with respect to space rotated according to the
    target coordinate system.
    """
    if target is None:
        target = mapping.coordinate_system
    if sampler is None:
        sampler = get_sampler(sampler)
    resampled_mapping = mapping.resample_to(
        mapping,
    )
    grid_spacing = target.coordinate_system.grid_spacing()
    n_dims = len(target.coordinate_system.spatial_shape)
    return stack_mappable_tensors(
        *(
            resampled_mapping.modify_sampler(
                sampler.derivative(spatial_dim=spatial_dim, limit_direction=limit_direction),
            ).sample_to(target)
            / grid_spacing[spatial_dim]
            for spatial_dim in range(n_dims)
        ),
        channel_index=-1,
    )
