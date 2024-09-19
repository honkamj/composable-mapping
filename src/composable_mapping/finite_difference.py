"""Calculate spatial derivatives with respect to the volumes"""

from itertools import product
from typing import Optional, Sequence, Union

from torch import Tensor, tensor

from .interface import IMaskedTensor
from .masked_tensor import MaskedTensor, stack_channels
from .util import num_spatial_dims
from .voxel_coordinate_system import ReferenceOption, VoxelCoordinateSystem

_OTHER_DIMS_TO_SHIFT = {
    "crop": 1,
    "crop_first": 1,
    "crop_last": 0,
    "average": 0.5,
    None: 0,
}
_OTHER_DIMS_TO_SHAPE_DIFFERENCE = {
    "crop": -2,
    "crop_first": -1,
    "crop_last": -1,
    "average": -1,
    None: 0,
}
_CENTRAL_TO_SHIFT = {
    True: 1.0,
    False: 0.5,
}
_CENTRAL_TO_SHAPE_DIFFERENCE = {
    True: -2,
    False: -1,
}


def update_coordinate_system_for_derivatives(
    coordinate_system: VoxelCoordinateSystem,
    spatial_dim: int,
    other_dims: Optional[str],
    central: bool,
) -> VoxelCoordinateSystem:
    """Update coordinate system to match the shape of the derivatives"""
    shifts = tuple(
        _OTHER_DIMS_TO_SHIFT[other_dims] if dim != spatial_dim else _CENTRAL_TO_SHIFT[central]
        for dim in range(len(coordinate_system.shape))
    )
    target_shape = tuple(
        (
            dim_size
            + (
                _OTHER_DIMS_TO_SHAPE_DIFFERENCE[other_dims]
                if dim != spatial_dim
                else _CENTRAL_TO_SHAPE_DIFFERENCE[central]
            )
        )
        for dim, dim_size in enumerate(coordinate_system.shape)
    )
    return coordinate_system.reformat(
        shape=target_shape, source_reference=shifts, target_reference=0
    )


def update_coordinate_system_for_jacobian_matrices(
    coordinate_system: VoxelCoordinateSystem,
    central: bool,
) -> VoxelCoordinateSystem:
    """Update coordinate system to match the shape of the Jacobian matrices"""
    target_shape = tuple(
        (dim_size + _CENTRAL_TO_SHAPE_DIFFERENCE[central]) for dim_size in coordinate_system.shape
    )
    return coordinate_system.reformat(
        shape=target_shape, source_reference=ReferenceOption("center")
    )


def estimate_spatial_derivatives(
    volume: IMaskedTensor,
    spatial_dim: int,
    spacing: Optional[Union[Tensor, float, int]] = None,
    other_dims: Optional[str] = None,
    central: bool = False,
) -> MaskedTensor:
    """Calculate spatial derivatives over a dimension estimated using finite differences

    Args:
        volume: Derivative is calculate over values of this mapping
            Tensor with shape (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            dim_1, ..., dim_{n_dims})
        spatial_dim: Dimension over which to compute the derivative,
            indexing starts from 0 and corresponds to dim_1 above.
        spacing: Spacing between voxels over the spatial_dim with shape (1 or batch_size,)
        n_channel_dims: Number of channel dimensions
        other_dims: If not given, the shape over other dimensions will not
                change. If given, must one of the following options:
            average: Other dimensions are averaged over two consequtive slices
                to obtain same shape difference as the dimension over which the
                derivative is computed. This can not be used if central == True.
            crop: Other dimensions are cropped to obtain same shape difference
                as the dimension over which the derivative is computed. If
                central == False, equals to the option crop_last.
            crop_first: Other dimensions are cropped to obtain same shape
                difference as the dimension over which the derivative is
                computed by cropping the first element. This can not be used if
                central == True.
            crop_last: Other dimensions are cropped to obtain same shape
                difference as the dimension over which the derivative is
                computed by cropping the last element. This can not be used if
                central == True.
        central: Whether to use central difference [f(x + 1)  - f(x - 1)] / 2 or not
            f(x + 1) - f(x)
        out: Save output to this Tensor

    Returns:
        if central and other_dims == "crop": Tensor with shape
            (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            dim_1 - 2, ..., dim_{n_dims} - 2)
        elif central and other_dims is None: Tensor with shape
            (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            dim_1, ..., dim_{spatial_dim} - 2, ..., dim_{n_dims})
        elif not central and other_dims not in (None, "crop_both"): Tensor with shape
            (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            dim_1 - 1, ..., dim_{n_dims} - 1)
        elif not central and other_dims is None: Tensor with shape
            (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
            dim_1, ..., dim_{spatial_dim} - 1, ..., dim_{n_dims})
    """
    if central and other_dims not in (None, "crop"):
        raise ValueError(f'Can not use central difference with option other_dims == "{other_dims}"')
    data, mask = volume.generate(generate_missing_mask=False, cast_mask=False)
    n_channel_dims = len(volume.channels_shape)
    batch_size = data.shape[0]
    if spacing is None:
        spacing = 1.0
    if isinstance(spacing, float) or isinstance(spacing, int):
        spacing = tensor(spacing, dtype=data.dtype, device=data.device)
    spacing = spacing.expand((batch_size,))[(...,) + (None,) * (data.ndim - 1)]
    n_spatial_dims = num_spatial_dims(data.ndim, n_channel_dims)
    if other_dims == "crop":
        other_crop = slice(1, -1) if central else slice(None, -1)
    elif other_dims == "crop_first":
        other_crop = slice(1, None)
    elif other_dims == "crop_last":
        other_crop = slice(None, -1)
    else:
        other_crop = slice(None)
    if central:
        front_crop = slice(2, None)
        back_crop = slice(None, -2)
    else:
        front_crop = slice(1, None)
        back_crop = slice(None, -1)
    front_cropping_slice = (...,) + tuple(
        front_crop if i == spatial_dim else other_crop for i in range(n_spatial_dims)
    )
    back_cropping_slice = (...,) + tuple(
        back_crop if i == spatial_dim else other_crop for i in range(n_spatial_dims)
    )
    derivatives = (data[front_cropping_slice] - data[back_cropping_slice]) / spacing
    if mask is not None:
        updated_mask: Optional[Tensor] = mask[front_cropping_slice] & mask[back_cropping_slice]
    else:
        updated_mask = None
    if central:
        derivatives = derivatives / 2
    elif other_dims == "average":
        summed_derivatives: Optional[Tensor] = None
        combined_mask: Optional[Tensor] = None
        for slice_parts in product((slice(1, None), slice(None, -1)), repeat=n_spatial_dims - 1):
            shifting_slice = (
                (...,) + slice_parts[:spatial_dim] + (slice(None),) + slice_parts[spatial_dim:]
            )
            if summed_derivatives is None:
                summed_derivatives = derivatives[shifting_slice]
            else:
                summed_derivatives = summed_derivatives + derivatives[shifting_slice]
            if mask is not None:
                if combined_mask is None:
                    combined_mask = mask[shifting_slice]
                else:
                    combined_mask = combined_mask & mask[shifting_slice]
        derivatives = summed_derivatives / 2 ** (n_spatial_dims - 1)
        updated_mask = combined_mask
    return MaskedTensor(derivatives, mask=updated_mask, n_channel_dims=n_channel_dims)


def estimate_spatial_jacobian_matrices(
    volume: IMaskedTensor,
    spacing: Optional[Union[Sequence[Union[float, int]], float, int, Tensor]] = None,
    central: bool = False,
) -> MaskedTensor:
    """Calculate local Jacobian matrices of a volume estimated using finite differences

    Args:
        volume: Tensor with shape (batch_size, n_dims, dim_1, ..., dim_{n_dims}),
            regularly sampled values of some mapping with the given spacing
        spacing: Voxel sizes along each dimension with shape ([batch_size, ]n_dims)
        n_channel_dims: Number of channel dimensions
        central: See option central of algorithm.finite_difference

    Returns:
        if central
            Tensor with shape (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
                n_dims, dim_1 - 2, ..., dim_{n_dims} - 2)
        else:
            Tensor with shape (batch_size, channel_dim_1, ..., channel_dim_{n_channel_dims},
                n_dims, dim_1 - 1, ..., dim_{n_dims} - 1)

    """
    data, mask = volume.generate(generate_missing_mask=False, cast_mask=False)
    n_channel_dims = len(volume.channels_shape)
    n_spatial_dims = len(volume.spatial_shape)
    if spacing is None:
        spacing = 1.0
    if not isinstance(spacing, Tensor):
        spacing = tensor(spacing, dtype=volume.dtype, device=volume.device)
    batch_size = volume.shape[0]
    spacing = spacing.expand((batch_size, n_spatial_dims))
    return stack_channels(
        *(
            estimate_spatial_derivatives(
                volume=MaskedTensor(data, mask, n_channel_dims=n_channel_dims),
                spatial_dim=dim,
                spacing=spacing[:, dim],
                other_dims={False: "average", True: "crop"}[central],
                central=central,
            )
            for dim in range(n_spatial_dims)
        ),
        channel_index=n_channel_dims,
    )
