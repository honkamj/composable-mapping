"""Utilities for mappable tensors."""

from typing import Optional

from torch import Tensor, cat, stack

from composable_mapping.util import combine_optional_masks, get_channel_dims

from .mappable_tensor import MappableTensor, PlainTensor


def concatenate_channels(
    *masked_tensors: MappableTensor, channel_index: int = 0
) -> "MappableTensor":
    """Concatenate masked tensors along the channel dimension

    Args:
        masked_tensors: Masked tensors to concatenate
        combine_mask: Combine masks of the masked tensors by multiplying,
            otherwise the mask of the first tensor is used.
        channel_index: Index of the channel dimension starting from the first
            channel dimension (second dimension of the tensor if batch dimension
            is present)
    """
    if not all(
        masked_tensor.n_channel_dims == masked_tensors[0].n_channel_dims
        for masked_tensor in masked_tensors[1:]
    ):
        raise ValueError("Lengths of channel shapes of masked tensors must be the same")
    n_channel_dims = len(masked_tensors[0].channels_shape)
    concatenation_dim = get_channel_dims(
        n_total_dims=len(masked_tensors[0].shape),
        n_channel_dims=n_channel_dims,
    )[channel_index]
    values = cat(
        [masked_tensor.generate_values() for masked_tensor in masked_tensors],
        dim=concatenation_dim,
    )
    mask: Optional[Tensor] = None
    for masked_tensor in masked_tensors:
        update_mask = masked_tensor.generate_mask(generate_missing_mask=False, cast_mask=False)
        mask = combine_optional_masks([mask, update_mask])
    return PlainTensor(
        values=values,
        mask=mask,
        n_channel_dims=n_channel_dims,
    )


def stack_channels(*masked_tensors: MappableTensor, channel_index: int = 0) -> "MappableTensor":
    """Concatenate masked tensors along the channel dimension

    Args:
        masked_tensors: Masked tensors to concatenate
        combine_mask: Combine masks of the masked tensors by multiplying,
            otherwise the mask of the first tensor is used.
        channel_index: Index of the channel dimension over which to stack
            starting from the first channel dimension (second dimension of the
            tensor if batch dimension is present)
    """
    if not all(
        masked_tensor.n_channel_dims == masked_tensors[0].n_channel_dims
        for masked_tensor in masked_tensors[1:]
    ):
        raise ValueError("Lengths of channel shapes of masked tensors must be the same")
    n_channel_dims = len(masked_tensors[0].channels_shape)
    channel_dims = get_channel_dims(
        n_total_dims=len(masked_tensors[0].shape),
        n_channel_dims=n_channel_dims,
    )
    stacking_dim = (channel_dims + (channel_dims[-1] + 1,))[channel_index]
    values = stack(
        [masked_tensor.generate_values() for masked_tensor in masked_tensors],
        dim=stacking_dim,
    )
    mask: Optional[Tensor] = None
    for masked_tensor in masked_tensors:
        update_mask = masked_tensor.generate_mask(generate_missing_mask=False, cast_mask=False)
        if update_mask is not None:
            update_mask = update_mask.unsqueeze(dim=stacking_dim)
            mask = combine_optional_masks([mask, update_mask])
    return PlainTensor(
        values=values,
        mask=mask,
        n_channel_dims=n_channel_dims + 1,
    )
