"""Tools for estimating spatial derivatives of a composable mappings."""

from typing import Optional

from .composable_mapping import GridComposableMapping, ICoordinateSystemContainer
from .mappable_tensor import MappableTensor, stack_mappable_tensors
from .sampler import ISampler, LimitDirection, get_sampler


def estimate_spatial_derivatives(
    mapping: GridComposableMapping,
    spatial_dim: int,
    target: Optional[ICoordinateSystemContainer] = None,
    limit_direction: LimitDirection = LimitDirection.average(),
    sampler: Optional[ISampler] = None,
) -> MappableTensor:
    """Estimate spatial derivatives of a grid composable mapping.

    Derivatives are computed with respect to space rotated according to the
    coordinate system of the target.

    Args:
        mapping: Grid composable mapping
        spatial_dim: Spatial dimension along which to compute the derivative
        target: Target locations at which to estimate the derivative.
        limit_direction: Direction in which to compute the derivative
        sampler: Sampler to use for the derivative estimation, e.g. LinearInterpolator
            corresponds to finite differences.

    Returns:
        MappableTensor with the estimated derivatives over spatial locations.
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
    limit_direction: LimitDirection = LimitDirection.average(),
    sampler: Optional[ISampler] = None,
) -> MappableTensor:
    """Estimate spatial Jacobian matrices of a grid composable mapping

    Jacobian matrices are computed with respect to space rotated according to the
    coordinate system of the target.

    Args:
        mapping: Grid composable mapping to estimate the Jacobian matrices for.
        target: Target locations at which to estimate the Jacobian matrices.
        limit_direction: Direction in which to compute the derivatives.
        sampler: Sampler to use for the derivative estimation, e.g. LinearInterpolator
            corresponds to finite differences.

    Returns:
        MappableTensor with the estimated Jacobian matrices over spatial locations.
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
