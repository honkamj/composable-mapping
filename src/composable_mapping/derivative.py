"""Tools for estimating spatial derivatives of a composable mappings."""

from typing import Callable, Optional, Sequence, Union

from torch import matmul

from .composable_mapping import GridComposableMapping, ICoordinateSystemContainer
from .mappable_tensor import MappableTensor, stack_mappable_tensors
from .sampler import DataFormat, ISampler, LimitDirection, get_sampler
from .util import (
    broadcast_tensors_in_parts,
    get_batch_shape,
    move_channels_first,
    move_channels_last,
    split_shape,
)


def estimate_spatial_jacobian_matrices(
    mapping: GridComposableMapping,
    target: Optional[ICoordinateSystemContainer] = None,
    limit_direction: Union[LimitDirection, Sequence[LimitDirection]] = LimitDirection.average(),
    sampler: Optional[ISampler] = None,
) -> MappableTensor:
    """Estimate spatial Jacobian matrices of a grid composable mapping.

    Estimation is done based on samples of the mapping at the grid defined by
    the coordinate system of the mapping.

    Args:
        mapping: Grid composable mapping to estimate the Jacobian matrices for.
        target: Target locations at which to estimate the Jacobian matrices.
        limit_direction: Direction in which to compute the derivatives, e.g. average
            and LinearInterpolator corresponds to central finite differences when
            estimated at the grid points.
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
        data_format=DataFormat.world_coordinates(),
    )
    n_dims = len(target.coordinate_system.spatial_shape)
    if isinstance(limit_direction, LimitDirection):
        limit_direction = [limit_direction] * n_dims
    sampled_jacobians = stack_mappable_tensors(
        *(
            resampled_mapping.modify_sampler(
                sampler.derivative(
                    spatial_dim=spatial_dim, limit_direction=limit_direction[spatial_dim]
                ),
            ).sample_to(target)
            for spatial_dim in range(n_dims)
        ),
        channel_index=-1,
    )
    jacobian_matrices, jacobians_mask = sampled_jacobians.generate()
    target_jacobian_matrix_shape = sampled_jacobians.channels_shape[:-1] + (
        len(target.coordinate_system.spatial_shape),
    )
    jacobian_matrices = jacobian_matrices.reshape(
        sampled_jacobians.batch_shape
        + (-1, sampled_jacobians.channels_shape[-1])
        + sampled_jacobians.spatial_shape
    )
    coordinate_system_affine_transformation = (
        target.coordinate_system.to_voxel_coordinates.transformation
    )
    coordinate_system_diagonal_affine_matrix = coordinate_system_affine_transformation.as_diagonal()
    if coordinate_system_diagonal_affine_matrix is None:
        coordinate_system_affine_matrix = coordinate_system_affine_transformation.as_matrix()
        jacobian_matrices, coordinate_system_affine_matrix = broadcast_tensors_in_parts(
            jacobian_matrices,
            coordinate_system_affine_matrix,
            broadcast_channels=False,
            n_channel_dims=(2, 2),
        )
        composed_jacobian_matrices = move_channels_first(
            matmul(
                move_channels_last(jacobian_matrices, n_channel_dims=2),
                move_channels_last(coordinate_system_affine_matrix, n_channel_dims=2)[
                    ..., :-1, :-1
                ],
            ),
            n_channel_dims=2,
        )
    else:
        jacobian_matrices, coordinate_system_diagonal = broadcast_tensors_in_parts(
            jacobian_matrices,
            coordinate_system_diagonal_affine_matrix.generate_diagonal(),
            broadcast_channels=False,
            n_channel_dims=(2, 1),
        )
        n_batch_dims = len(get_batch_shape(coordinate_system_diagonal.shape, n_channel_dims=1))
        composed_jacobian_matrices = (
            jacobian_matrices * coordinate_system_diagonal[n_batch_dims * (slice(None),) + (None,)]
        )
    batch_shape, _, spatial_shape = split_shape(jacobian_matrices.shape, n_channel_dims=2)
    composed_jacobian_matrices = composed_jacobian_matrices.reshape(
        batch_shape + target_jacobian_matrix_shape + spatial_shape
    )
    return MappableTensor(
        composed_jacobian_matrices,
        mask=jacobians_mask,
        n_channel_dims=sampled_jacobians.n_channel_dims,
    )
