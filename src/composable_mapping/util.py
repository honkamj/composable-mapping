"""Utility functions"""

from itertools import repeat
from typing import Iterable, Optional, Sequence, Tuple, TypeVar, Union, cast

from torch import Tensor, broadcast_shapes

T = TypeVar("T")


def ceildiv(denominator: Union[int, float], numerator: Union[int, float]) -> Union[int, float]:
    """Ceil integer division"""
    return -(denominator // -numerator)


def optional_add(addable_1: Optional[T], addable_2: Optional[T]) -> Optional[T]:
    """Optional add"""
    if addable_1 is None:
        return addable_2
    if addable_2 is None:
        return addable_1
    added = addable_1 + addable_2  # type: ignore
    return cast(T, added)


def get_channel_dims(n_total_dims: int, n_channel_dims: int) -> Tuple[int, ...]:
    """Returns indices for channel dimensions"""
    if n_total_dims < n_channel_dims:
        raise RuntimeError(
            "Number of channel dimensions can not be larger than total number of dimensions"
        )
    if n_total_dims == n_channel_dims:
        return tuple(range(n_total_dims))
    return tuple(range(1, n_channel_dims + 1))


def get_spatial_dims(n_total_dims: int, n_channel_dims: int) -> Tuple[int, ...]:
    """Returns indices for spatial dimensions"""
    if n_total_dims < n_channel_dims:
        raise RuntimeError(
            "Number of channel dimensions can not be larger than total number of dimensions"
        )
    last_channel_dim = get_channel_dims(n_total_dims, n_channel_dims)[-1]
    return tuple(range(last_channel_dim + 1, n_total_dims))


def get_batch_dims(n_total_dims: int, n_channel_dims: int) -> Tuple[int, ...]:
    """Returns indices for batch dimensions"""
    if n_total_dims < n_channel_dims:
        raise RuntimeError(
            "Number of channel dimensions can not be larger than total number of dimensions"
        )
    first_channel_dim = get_channel_dims(n_total_dims, n_channel_dims)[0]
    return tuple(range(first_channel_dim))


def move_channels_first(tensor: Tensor, n_channel_dims: int = 1) -> Tensor:
    """Move channel dimensions first

    Args:
        tensor: Tensor with shape (batch_size, *, channel_1, ..., channel_{n_channel_dims})
        n_channel_dims: Number of channel dimensions

    Returns: Tensor with shape (batch_size, channel_1, ..., channel_{n_channel_dims}, *)
    """
    return tensor.moveaxis(
        tuple(range(-n_channel_dims, 0)),
        get_channel_dims(tensor.ndim, n_channel_dims=n_channel_dims),
    )


def move_channels_last(tensor: Tensor, n_channel_dims: int = 1) -> Tensor:
    """Move channel dimensions last

    Args:
        tensor: Tensor with shape (batch_size, channel_1, ..., channel_{n_channel_dims}, *)
        n_channel_dims: Number of channel dimensions

    Returns: Tensor with shape (batch_size, *, channel_1, ..., channel_{n_channel_dims})
    """
    return tensor.moveaxis(
        get_channel_dims(tensor.ndim, n_channel_dims=n_channel_dims),
        list(range(-n_channel_dims, 0)),
    )


def _n_channel_dims_to_iterable(n_channel_dims: Union[int, Iterable[int]]) -> Iterable[int]:
    if isinstance(n_channel_dims, int):
        return repeat(n_channel_dims)
    else:
        return n_channel_dims


def broadcast_shapes_in_parts_splitted(
    *shapes: Sequence[int],
    n_channel_dims: Union[int, Iterable[int]] = 1,
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
) -> Tuple[Optional[Tuple[int, ...]], Optional[Tuple[int, ...]], Optional[Tuple[int, ...]]]:
    """Broadcasts batch dimension, channel dimensions and spatial dimensions separately

    Args:
        shapes: Shapes to broadcast
        n_channel_dims: Number of channel dims for each shape, if
            integer is given, same number will be used for all shapes. Channel dimensions
            are assumed to come after the batch dimension.
    """
    channel_dims_iterable = _n_channel_dims_to_iterable(n_channel_dims)
    splitted_shapes = [
        split_shape(shape, individual_n_channel_dims)
        for shape, individual_n_channel_dims in zip(shapes, channel_dims_iterable)
    ]
    if broadcast_batch:
        broadcasted_batch_shape: Optional[Tuple[int, ...]] = broadcast_shapes(
            *(shape[0] for shape in splitted_shapes)
        )
    else:
        broadcasted_batch_shape = None
    if broadcast_channels:
        broadcasted_channel_shape: Optional[Tuple[int, ...]] = broadcast_shapes(
            *(shape[1] for shape in splitted_shapes)
        )
    else:
        broadcasted_channel_shape = None
    if broadcast_spatial:
        broadcasted_spatial_shape: Optional[Tuple[int, ...]] = broadcast_shapes(
            *(shape[2] for shape in splitted_shapes)
        )
    else:
        broadcasted_spatial_shape = None
    return broadcasted_batch_shape, broadcasted_channel_shape, broadcasted_spatial_shape


def broadcast_shapes_in_parts(
    *shapes: Sequence[int],
    n_channel_dims: Union[int, Iterable[int]] = 1,
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
) -> Sequence[Tuple[int, ...]]:
    """Broadcasts shapes spatially"""
    batch_shape, channel_shape, spatial_shape = broadcast_shapes_in_parts_splitted(
        *shapes,
        n_channel_dims=n_channel_dims,
        broadcast_batch=broadcast_batch,
        broadcast_channels=broadcast_channels,
        broadcast_spatial=broadcast_spatial,
    )
    output_shapes = []
    for shape, individual_n_channel_dims in zip(
        shapes, _n_channel_dims_to_iterable(n_channel_dims)
    ):
        target_batch_shape, target_channel_shape, target_spatial_shape = split_shape(
            shape, individual_n_channel_dims
        )
        if batch_shape is not None:
            target_batch_shape = batch_shape
        if channel_shape is not None:
            target_channel_shape = channel_shape
        if spatial_shape is not None:
            target_spatial_shape = spatial_shape
        output_shapes.append(target_batch_shape + target_channel_shape + target_spatial_shape)
    return output_shapes


def broadcast_shapes_in_parts_to_single_shape(
    *shapes: Sequence[int],
    n_channel_dims: Union[int, Iterable[int]] = 1,
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
) -> Tuple[int, ...]:
    """Broadcasts shapes spatially to single shape

    Raises:
        RuntimeError: If the shapes do not broadcast to the same shape
    """
    broadcasted_shapes = broadcast_shapes_in_parts(
        *shapes,
        n_channel_dims=n_channel_dims,
        broadcast_batch=broadcast_batch,
        broadcast_channels=broadcast_channels,
        broadcast_spatial=broadcast_spatial,
    )
    if len(set(broadcasted_shapes)) != 1:
        raise RuntimeError("Shapes do not broadcast to the same shape")
    return broadcasted_shapes[0]


def broadcast_to_in_parts(
    tensor: Tensor,
    batch_shape: Optional[Sequence[int]] = None,
    channels_shape: Optional[Sequence[int]] = None,
    spatial_shape: Optional[Sequence[int]] = None,
    n_channel_dims: int = 1,
) -> Tensor:
    """Broadcasts tensor to given shapes, if None, the part of the shape is not
    broadcasted"""
    initial_batch_shape, initial_channels_shape, initial_spatial_shape = split_shape(
        tensor.shape, n_channel_dims=n_channel_dims
    )
    if batch_shape is not None:
        if len(batch_shape) < len(initial_batch_shape):
            initial_batch_shape = initial_batch_shape[-len(batch_shape) :]
        else:
            initial_batch_shape = (1,) * (
                len(batch_shape) - len(initial_batch_shape)
            ) + initial_batch_shape
        tensor = tensor.reshape(
            initial_batch_shape + initial_channels_shape + initial_spatial_shape
        )
    else:
        batch_shape = initial_batch_shape
    if channels_shape is not None:
        if len(channels_shape) < len(initial_channels_shape):
            initial_channels_shape = initial_channels_shape[-len(channels_shape) :]
        else:
            initial_channels_shape = (1,) * (
                len(channels_shape) - len(initial_channels_shape)
            ) + initial_channels_shape
        tensor = tensor.reshape(
            initial_batch_shape + initial_channels_shape + initial_spatial_shape
        )
    else:
        channels_shape = initial_channels_shape
    if spatial_shape is not None:
        if len(spatial_shape) < len(initial_spatial_shape):
            initial_spatial_shape = initial_spatial_shape[-len(spatial_shape) :]
        else:
            initial_spatial_shape = (1,) * (
                len(spatial_shape) - len(initial_spatial_shape)
            ) + initial_spatial_shape
        tensor = tensor.reshape(
            initial_batch_shape + initial_channels_shape + initial_spatial_shape
        )
    else:
        spatial_shape = initial_spatial_shape
    return tensor.broadcast_to(tuple(batch_shape) + tuple(channels_shape) + tuple(spatial_shape))


def broadcast_tensors_in_parts(
    *tensors: Tensor,
    broadcast_batch: bool = True,
    broadcast_channels: bool = True,
    broadcast_spatial: bool = True,
    n_channel_dims: Union[int, Iterable[int]] = 1,
) -> Tuple[Tensor, ...]:
    """Broadcasts tensors spatially"""
    shapes = [tensor.shape for tensor in tensors]
    broadcasted_batch_shape, broadcasted_channel_shape, broadcasted_spatial_shape = (
        broadcast_shapes_in_parts_splitted(
            *shapes,
            n_channel_dims=n_channel_dims,
            broadcast_batch=broadcast_batch,
            broadcast_channels=broadcast_channels,
            broadcast_spatial=broadcast_spatial,
        )
    )
    return tuple(
        broadcast_to_in_parts(
            tensor,
            batch_shape=broadcasted_batch_shape,
            channels_shape=broadcasted_channel_shape,
            spatial_shape=broadcasted_spatial_shape,
            n_channel_dims=individual_n_channel_dims,
        )
        for tensor, individual_n_channel_dims in zip(
            tensors, _n_channel_dims_to_iterable(n_channel_dims)
        )
    )


def split_shape(
    shape: Sequence[int], n_channel_dims: int
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Splits shape into batch, channel and spatial dimensions"""
    first_channel_dim = get_channel_dims(len(shape), n_channel_dims)[0]
    return (
        tuple(shape[:first_channel_dim]),
        tuple(shape[first_channel_dim : first_channel_dim + n_channel_dims]),
        tuple(shape[first_channel_dim + n_channel_dims :]),
    )


def get_channels_shape(shape: Sequence[int], n_channel_dims: int) -> Tuple[int, ...]:
    """Returns shape of the channel dimensions"""
    first_channel_dim = get_channel_dims(len(shape), n_channel_dims)[0]
    return tuple(shape[first_channel_dim : first_channel_dim + n_channel_dims])


def get_spatial_shape(shape: Sequence[int], n_channel_dims: int) -> Tuple[int, ...]:
    """Returns shape of the spatial dimensions"""
    last_channel_dim = get_channel_dims(len(shape), n_channel_dims)[-1]
    return tuple(shape[last_channel_dim + 1 :])


def get_batch_shape(shape: Sequence[int], n_channel_dims: int) -> Tuple[int, ...]:
    """Returns size of the batch dimensions"""
    first_channel_dim = get_channel_dims(len(shape), n_channel_dims)[0]
    return tuple(shape[:first_channel_dim])


def reduce_channel_shape_to_ones(shape: Sequence[int], n_channel_dims: int) -> Tuple[int, ...]:
    """Reduces channel shape to ones

    E.g. (3, 5, 4, 4) with n_channel_dims = 1 returns
    (3, 1, 4, 4)
    """
    if len(shape) <= n_channel_dims:
        return (1,) * len(shape)
    return tuple(shape[:1]) + (1,) * n_channel_dims + tuple(shape[1 + n_channel_dims :])


def num_spatial_dims(n_total_dims: int, n_channel_dims: int) -> int:
    """Returns amount of spatial dimensions"""
    if n_total_dims < n_channel_dims:
        raise RuntimeError("Number of channel dimensions do not match")
    if n_total_dims <= n_channel_dims + 1:
        return 0
    return n_total_dims - n_channel_dims - 1


def combine_optional_masks(
    masks: Sequence[Optional[Tensor]],
    n_channel_dims: Union[int, Iterable[int]] = 1,
) -> Optional[Tensor]:
    """Combine optional masks"""
    broadcasted_masks = broadcast_tensors_in_parts(
        *(mask for mask in masks if mask is not None), n_channel_dims=n_channel_dims
    )
    combined_mask: Optional[Tensor] = None
    for mask in broadcasted_masks:
        if mask is not None:
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask & mask
    return combined_mask
