"""Utility functions"""

from functools import wraps
from inspect import signature
from itertools import repeat
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from torch import Tensor, broadcast_shapes, broadcast_to
from torch.jit import script


@script
def move_channels_first(tensor: Tensor, num_channel_dims: int = 1) -> Tensor:
    """Move channel dimensions first

    Args:
        tensor: Tensor with shape (batch_size, *, channel_1, ..., channel_{num_channel_dims})
        num_channel_dims: Number of channel dimensions

    Returns: Tensor with shape (batch_size, channel_1, ..., channel_{num_channel_dims}, *)
    """
    if tensor.ndim == num_channel_dims:
        return tensor
    return tensor.permute(
        [0] + list(range(-num_channel_dims, 0)) + list(range(1, tensor.ndim - num_channel_dims))
    )


@script
def move_channels_last(tensor: Tensor, num_channel_dims: int = 1) -> Tensor:
    """Move channel dimensions last

    Args:
        tensor: Tensor with shape (batch_size, channel_1, ..., channel_{num_channel_dims}, *)
        num_channel_dims: Number of channel dimensions

    Returns: Tensor with shape (batch_size, *, channel_1, ..., channel_{num_channel_dims})
    """
    if tensor.ndim == num_channel_dims:
        return tensor
    return tensor.permute(
        [0] + list(range(num_channel_dims + 1, tensor.ndim)) + list(range(1, num_channel_dims + 1))
    )


@script
def merge_batch_dimensions(tensor: Tensor, num_channel_dims: int = 1) -> Tuple[Tensor, List[int]]:
    """Merges batch dimensions

    Args:
        tensor: Tensor with shape (..., channel_1, ..., channel_{num_channel_dims})

    Returns: Tensor with shape (batch_size, channel_1, ..., channel_{num_channel_dims})
    """
    batch_dimensions_shape = list(tensor.shape[:-num_channel_dims])
    if num_channel_dims == 0:
        channels_shape: List[int] = []
    else:
        channels_shape = list(tensor.shape[-num_channel_dims:])
    return tensor.reshape([-1] + channels_shape), batch_dimensions_shape


@script
def unmerge_batch_dimensions(
    tensor: Tensor, batch_dimensions_shape: List[int], num_channel_dims: int = 1
) -> Tensor:
    """Unmerges batch dimensions

    Args:
        tensor: Tensor with shape (batch_size, channel_1, ..., channel_{num_channel_dims})

    Returns: Tensor with shape (*batch_dimensions_shape, channel_1, ..., channel_{num_channel_dims})
    """
    if num_channel_dims == 0:
        channels_shape: List[int] = []
    else:
        channels_shape = list(tensor.shape[-num_channel_dims:])
    return tensor.view(batch_dimensions_shape + channels_shape)


class BatchDimensionMerger:
    """Merges or unmerges batch dimensions

    Unmerging assumes broadcasted batch shapes from merge.
    """

    def __init__(self) -> None:
        self._original_batch_shape: Optional[Sequence[int]] = None

    def merge(self, tensor: Tensor, num_channel_dims: int) -> Tensor:
        """Merge batch dimensions"""
        batch_merged_tensor, batch_shape = merge_batch_dimensions(tensor, num_channel_dims)
        if self._original_batch_shape is None:
            self._original_batch_shape = batch_shape
        else:
            self._original_batch_shape = broadcast_shapes(self._original_batch_shape, batch_shape)
        return batch_merged_tensor

    def unmerge(self, tensor: Tensor, num_channel_dims: int) -> Tensor:
        """Unmerge batch dimensions"""
        if self._original_batch_shape is None:
            raise RuntimeError("BatchDimensionMerger::merge must be called first!")
        return unmerge_batch_dimensions(tensor, self._original_batch_shape, num_channel_dims)


def merged_batch_dimensions(
    num_input_non_batch_dims: Optional[Union[int, Mapping[str, int]]] = 1,
    num_return_non_batch_dims: Optional[Union[int, Sequence[int]]] = 1,
):
    """Function wrapper that merges batch dimensions into one
    dimension for input and then splits them back for output

    Channel dimensions are assumed to be the last ones. See the class
    FunctionChannelStructureModifier for argument specification
    """

    def _wrapper_func(func):
        @wraps(func)
        def _modified_func(*args, **kwargs):
            batch_dimension_merger = BatchDimensionMerger()
            modifier = FunctionDimensionStructureModifier(
                num_input_channel_dims=num_input_non_batch_dims,
                num_return_channel_dims=num_return_non_batch_dims,
                input_modifier=batch_dimension_merger.merge,
                output_modifier=batch_dimension_merger.unmerge,
            )
            return modifier.modified_call(func, *args, **kwargs)

        return _modified_func

    return _wrapper_func


class FunctionDimensionStructureModifier:
    """Modifies function dimension structure for arguments and output
    based on given modifiers

    Arguments:
        num_return_channel_dims: How many channel dimensions do the
            returned Tensors have. If None, channels will not be modified
            for the returned tensor
        num_input_channel_dims: How many channel dimensions do the
            inputs have. Can be defined as mapping based on the function
            argument names. If None, channels will not be modified
            for the inputs.
        input_modifier: Modifies the input tensors, the second argument
            is number of channels that the input has.
        output_modifier: Modifies the output tensors, the second argument
            is number of channels that the output has.
    """

    def __init__(
        self,
        num_input_channel_dims: Optional[Union[int, Mapping[str, int]]] = 1,
        num_return_channel_dims: Optional[Union[int, Sequence[int]]] = 1,
        input_modifier: Optional[Callable[[Tensor, int], Tensor]] = None,
        output_modifier: Optional[Callable[[Tensor, int], Tensor]] = None,
    ) -> None:
        self._num_input_channel_dims = num_input_channel_dims
        self._num_return_channel_dims = num_return_channel_dims
        self._input_modifier: Callable[[Tensor, int], Tensor] = (
            input_modifier
            if input_modifier is not None
            else lambda input_tensor, _num_channels: input_tensor
        )
        self._output_modifier: Callable[[Tensor, int], Tensor] = (
            output_modifier
            if output_modifier is not None
            else lambda input_tensor, _num_channels: input_tensor
        )

    def modified_call(self, func: Callable, *args, **kwargs) -> Any:
        """Calls the function with modified args and return values"""
        args_num_channels, kwargs_num_channels = self._extract_num_channels_for_args_and_kwargs(
            func=func, args=args, kwargs=kwargs
        )
        modified_args = [
            (
                self._input_modifier(arg_value, num_channel_dims)
                if num_channel_dims is not None
                else arg_value
            )
            for arg_value, num_channel_dims in args_num_channels
        ]
        modified_kwargs = {
            kwarg_name: (
                self._input_modifier(kwarg_value, num_channel_dims)
                if num_channel_dims is not None
                else kwarg_value
            )
            for kwarg_name, (kwarg_value, num_channel_dims) in kwargs_num_channels.items()
        }
        return_values = func(*modified_args, **modified_kwargs)
        return self._modify_return_values(return_values)

    def _modify_return_values(self, return_values: Any) -> Any:
        possible_sequence_types = (tuple, list)
        sequence_types_boolean_array = [
            isinstance(return_values, sequence_type) for sequence_type in possible_sequence_types
        ]
        if any(sequence_types_boolean_array):
            return_sequence_type = possible_sequence_types[sequence_types_boolean_array.index(True)]
            num_return_channel_dims = (
                self._num_return_channel_dims
                if isinstance(self._num_return_channel_dims, Sequence)
                else repeat(self._num_return_channel_dims)
            )
            return return_sequence_type(
                (
                    self._output_modifier(return_value, num_channels)
                    if (isinstance(return_value, Tensor) and num_channels is not None)
                    else return_value
                )
                for return_value, num_channels in zip(return_values, num_return_channel_dims)
            )
        if isinstance(return_values, Tensor) and self._num_return_channel_dims is not None:
            if not isinstance(self._num_return_channel_dims, int):
                raise ValueError("Invalid type for specifying number of return channels")
            return self._output_modifier(return_values, self._num_return_channel_dims)
        return return_values

    def _extract_num_channels_for_args_and_kwargs(
        self, func: Callable, args, kwargs
    ) -> Tuple[Sequence[Tuple[Any, Optional[int]]], Mapping[str, Tuple[Any, Optional[int]]]]:
        args_num_channels: List[Tuple[Any, Optional[int]]] = []
        kwargs_num_channels: Dict[str, Tuple[Any, Optional[int]]] = {}
        if isinstance(self._num_input_channel_dims, int) or self._num_input_channel_dims is None:
            for arg_value in args:
                arg_num_channels = (
                    self._num_input_channel_dims if isinstance(arg_value, Tensor) else None
                )
                args_num_channels.append((arg_value, arg_num_channels))
            for kwarg_name, kwarg_value in kwargs.items():
                kwarg_num_channels = (
                    self._num_input_channel_dims if isinstance(kwarg_value, Tensor) else None
                )
                kwargs_num_channels[kwarg_name] = (kwarg_value, kwarg_num_channels)
        elif isinstance(self._num_input_channel_dims, Mapping):
            arguments = signature(func).bind(*args, **kwargs).arguments
            for arg_name, arg_value in arguments.items():
                if arg_name in self._num_input_channel_dims:
                    if isinstance(arg_value, Tensor):
                        kwargs_num_channels[arg_name] = (
                            arg_value,
                            self._num_input_channel_dims[arg_name],
                        )
                    else:
                        kwargs_num_channels[arg_name] = (arg_value, None)
                else:
                    kwargs_num_channels[arg_name] = (arg_value, None)
        else:
            raise ValueError("Invalid type for specifying number of input channels")
        return args_num_channels, kwargs_num_channels


def channels_last(
    num_input_channel_dims: Optional[Union[int, Mapping[str, int]]] = 1,
    num_return_channel_dims: Optional[Union[int, Sequence[int]]] = 1,
):
    """Function wrapper that moves channel dimensions to last for the inputs and
    then back for the return value

    Assumes that the first dimension is batch dimension. See the class
    FunctionChannelStructureModifier for argument specification
    """

    def _wrapper_func(func):
        @wraps(func)
        def _modified_func(*args, **kwargs):
            if "channels_first" in kwargs:
                channels_first = kwargs["channels_first"]
                del kwargs["channels_first"]
            else:
                channels_first = True
            if channels_first:
                modifier = FunctionDimensionStructureModifier(
                    num_input_channel_dims=num_input_channel_dims,
                    num_return_channel_dims=num_return_channel_dims,
                    input_modifier=move_channels_last,
                    output_modifier=move_channels_first,
                )
                return modifier.modified_call(func, *args, **kwargs)
            return func(*args, **kwargs)

        return _modified_func

    return _wrapper_func


def _anchor_dim(n_dims: int, channel_dims: int) -> int:
    return min(channel_dims, n_dims - 1)


def _num_spatial_dimensions(n_dims: int, n_channel_dims: int) -> int:
    return max(n_dims - n_channel_dims - 1, 0)


def _channel_dims_to_iterable(n_channel_dims: Union[int, Iterable[int]]) -> Iterable[int]:
    if isinstance(n_channel_dims, int):
        return repeat(n_channel_dims)
    else:
        return n_channel_dims


def broadcast_shapes_around_channel_dims(
    shapes: Iterable[Sequence[int]], n_channel_dims: Union[int, Iterable[int]] = 1
) -> Sequence[Sequence[int]]:
    """Broadcasts shapes such that they match on other dimensions except the
    channel dimension (or dimensions if n_channel_dims > 1)

    The channel dimensions are assumed to come after the batch dimension. The dimensions
    after the channel dimensions are usually spatial dimensions.

    If spatial dimensions differ, they are broadcasted starting from the last
    dimension.

    Useful for broadcasting to spatial data.

    Args:
        shapes: Shapes to broadcast
        n_channel_dims: Number of channel dims for each shape, if
            integer is given, same number will be used for all shapes.

    E.g.:
        - Shapes (3, 2) and (2,) will be broadcasted to (3, 2)
        - Shapes (3, 2, 7, 5) and (2,) will be broadcasted to (3, 2, 7, 5)
        - Shapes (5, 3, 2) and (2,) will be broadcasted to (5, 3, 2) and (5, 2) if
            n_channel_dims = (2, 1)

    See tests for more examples.
    """
    channel_dims_iterable = _channel_dims_to_iterable(n_channel_dims)
    max_spatial_dimensions = max(
        _num_spatial_dimensions(len(shape), channel_dims)
        for shape, channel_dims in zip(shapes, channel_dims_iterable)
    )
    batch_shape: Tuple[int, ...] = tuple()
    spatial_shape: Tuple[int, ...] = tuple()
    dimesion_added_shapes = []
    for shape, channel_dims in zip(shapes, channel_dims_iterable):
        shape = tuple(shape)
        anchor_dim = _anchor_dim(len(shape), channel_dims)
        if len(shape) > channel_dims:
            batch_shape = broadcast_shapes(batch_shape, shape[:1])
            spatial_shape = broadcast_shapes(spatial_shape, shape[anchor_dim + 1 :])
        dims_to_add = max_spatial_dimensions - _num_spatial_dimensions(len(shape), channel_dims)
        dimesion_added_shapes.append(
            shape[: anchor_dim + 1] + (1,) * dims_to_add + shape[anchor_dim + 1 :]
        )
    broadcasted_shapes = [
        broadcast_shapes(shape, batch_shape + (1,) * channel_dims + spatial_shape)
        for shape, channel_dims in zip(dimesion_added_shapes, channel_dims_iterable)
    ]
    return broadcasted_shapes


def broadcast_to_shape_around_channel_dims(
    tensor: Tensor, shape: Sequence[int], n_channel_dims: int
) -> Tensor:
    """Broadcasts tensor to shape based around the channel dimension (or dimensions)"""
    anchor_dim = _anchor_dim(tensor.ndim, n_channel_dims)
    dims_to_add = _num_spatial_dimensions(len(shape), n_channel_dims) - _num_spatial_dimensions(
        tensor.ndim, n_channel_dims
    )
    new_shape = tensor.shape[: anchor_dim + 1] + (1,) * dims_to_add + tensor.shape[anchor_dim + 1 :]
    if new_shape:
        tensor = tensor.view(new_shape)
    return broadcast_to(tensor, tuple(shape))


def broadcast_tensors_around_channel_dims(
    tensors: Iterable[Tensor], n_channel_dims: Union[int, Iterable[int]] = 1
) -> Sequence[Tensor]:
    """Broadcasts shapes first to left and then to right

    See broadcast_shapes_by_channel_dims for more info.

    Args:
        tensors: Tensors to broadcast
        n_channel_dims: Number of leading dims for each shape, if
            integer is given, same number will be used for all shapes.
    """
    target_shapes = broadcast_shapes_around_channel_dims(
        [tensor.shape for tensor in tensors], n_channel_dims
    )
    channel_dims_iterable = _channel_dims_to_iterable(n_channel_dims)
    return [
        broadcast_to_shape_around_channel_dims(tensor, shape, channel_dims)
        for tensor, shape, channel_dims in zip(tensors, target_shapes, channel_dims_iterable)
    ]


@script
def index_by_channel_dims(
    n_total_dims: int,
    channel_dim_index: int,
    n_channel_dims: int,
    inclusive_upper_bound: bool = False,
) -> int:
    """Returns index in orignal tensor based on channel dim index

    E.g. (3, 5, 4, 4) with channel_dim_index = 0, n_channel_dims = 1 returns 1
    """
    if n_total_dims < n_channel_dims:
        raise RuntimeError("Number of channel dimensions do not match")
    if channel_dim_index < 0:
        channel_dim_index += n_channel_dims
    if (
        channel_dim_index < 0
        or (channel_dim_index >= n_channel_dims and not inclusive_upper_bound)
        or channel_dim_index > n_total_dims
    ):
        raise RuntimeError("Invalid channel dimension index")
    if n_total_dims == n_channel_dims:
        return channel_dim_index
    return channel_dim_index + 1


@script
def get_other_than_channel_dims(n_total_dims: int, n_channel_dims: int) -> List[int]:
    """Returns indices for dimensions other than channel dimensions"""
    first_channel_dim_index = index_by_channel_dims(n_total_dims, 0, n_channel_dims=n_channel_dims)
    channel_dims = list(range(first_channel_dim_index, first_channel_dim_index + n_channel_dims))
    dims: List[int] = []
    for dim in range(n_total_dims):
        if dim not in channel_dims:
            dims.append(dim)
    return dims


def reduce_channel_shape_to_ones(shape: Sequence[int], n_channel_dims: int) -> Tuple[int, ...]:
    """Reduces channel shape to ones

    E.g. (3, 5, 4, 4) with n_channel_dims = 1 returns
    (3, 1, 4, 4)
    """
    if len(shape) <= n_channel_dims:
        return (1,) * len(shape)
    return tuple(shape[:1]) + (1,) * n_channel_dims + tuple(shape[1 + n_channel_dims :])


@script
def num_spatial_dims(n_total_dims: int, n_channel_dims: int) -> int:
    """Returns amount of spatial dimensions"""
    if n_total_dims < n_channel_dims:
        raise RuntimeError("Number of channel dimensions do not match")
    if n_total_dims <= n_channel_dims + 1:
        return 0
    return n_total_dims - n_channel_dims - 1


@script
def combine_optional_masks(masks: List[Optional[Tensor]]) -> Optional[Tensor]:
    """Combine optional masks"""
    combined_mask: Optional[Tensor] = None
    for mask in masks:
        if mask is not None:
            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = combined_mask & mask
    return combined_mask
