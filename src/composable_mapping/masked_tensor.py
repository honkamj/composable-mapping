"""Masked tensors"""

from typing import Literal, Mapping, Optional, Sequence, Tuple, Union, overload

from torch import Tensor, allclose
from torch import any as torch_any
from torch import bool as torch_bool
from torch import cat
from torch import device as torch_device
from torch import diag, diagonal
from torch import dtype as torch_dtype
from torch import get_default_dtype
from torch import int32 as torch_int32
from torch import ones
from torch import round as torch_round
from torch import stack, tensor

from .affine import IdentityAffineTransformation
from .base import BaseTensorLikeWrapper
from .dense_deformation import generate_voxel_coordinate_grid
from .interface import IAffineTransformation, IMaskedTensor, ITensorLike
from .util import (
    combine_optional_masks,
    get_other_than_channel_dims,
    index_by_channel_dims,
    reduce_channel_shape_to_ones,
)


def concatenate_channels(*masked_tensors: IMaskedTensor, channel_index: int = 0) -> "MaskedTensor":
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
        len(masked_tensor.channels_shape) == len(masked_tensors[0].channels_shape)
        for masked_tensor in masked_tensors
    ):
        raise ValueError("Lengths of channel shapes of masked tensors must be the same")
    n_channel_dims = len(masked_tensors[0].channels_shape)
    concatenation_index = index_by_channel_dims(
        n_total_dims=len(masked_tensors[0].shape),
        channel_dim_index=channel_index,
        n_channel_dims=n_channel_dims,
    )
    values = cat(
        [masked_tensor.generate_values() for masked_tensor in masked_tensors],
        dim=concatenation_index,
    )
    mask: Optional[Tensor] = None
    for masked_tensor in masked_tensors:
        update_mask = masked_tensor.generate_mask(generate_missing_mask=False, cast_mask=False)
        mask = combine_optional_masks([mask, update_mask])
    return MaskedTensor(
        values=values,
        mask=mask,
        n_channel_dims=n_channel_dims,
    )


def stack_channels(*masked_tensors: IMaskedTensor, channel_index: int = 0) -> "MaskedTensor":
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
        len(masked_tensor.channels_shape) == len(masked_tensors[0].channels_shape)
        for masked_tensor in masked_tensors
    ):
        raise ValueError("Lengths of channel shapes of masked tensors must be the same")
    n_channel_dims = len(masked_tensors[0].channels_shape)
    stack_index = index_by_channel_dims(
        n_total_dims=len(masked_tensors[0].shape),
        channel_dim_index=channel_index,
        n_channel_dims=n_channel_dims,
        inclusive_upper_bound=True,
    )
    values = stack(
        [masked_tensor.generate_values() for masked_tensor in masked_tensors],
        dim=stack_index,
    )
    mask: Optional[Tensor] = None
    for masked_tensor in masked_tensors:
        update_mask = masked_tensor.generate_mask(generate_missing_mask=False, cast_mask=False)
        if update_mask is not None:
            update_mask = update_mask.unsqueeze(dim=stack_index)
            mask = combine_optional_masks([mask, update_mask])
    return MaskedTensor(
        values=values,
        mask=mask,
        n_channel_dims=n_channel_dims + 1,
    )


class MaskedTensor(IMaskedTensor, BaseTensorLikeWrapper):
    """Masked tensor

    Arguments:
        values: Tensor with shape (batch_size, *channel_dims, *spatial_dims)
        mask: Tensor with shape (batch_size, 1, *spatial_dims)
    """

    def __init__(
        self,
        values: Tensor,
        mask: Optional[Tensor] = None,
        n_channel_dims: int = 1,
        affine_transformation: Optional[IAffineTransformation] = None,
    ) -> None:
        self._values = values
        self._mask = mask
        if mask is not None and mask.dtype != torch_bool:
            raise ValueError("Mask must be a boolean tensor")
        self._n_channel_dims = n_channel_dims
        self._affine_transformation: IAffineTransformation = (
            IdentityAffineTransformation(
                self.channels_shape[0], dtype=values.dtype, device=values.device
            )
            if affine_transformation is None
            else affine_transformation
        )

    @property
    def shape(self) -> Sequence[int]:
        return self._values.shape

    @property
    def spatial_shape(self) -> Sequence[int]:
        first_spatial_dim = index_by_channel_dims(self._values.ndim, -1, self._n_channel_dims) + 1
        return self._values.shape[first_spatial_dim:]

    @property
    def channels_shape(self) -> Sequence[int]:
        first_channel_dim = index_by_channel_dims(self._values.ndim, 0, self._n_channel_dims)
        return self._values.shape[first_channel_dim : first_channel_dim + self._n_channel_dims]

    @overload
    def generate(
        self,
        generate_missing_mask: Literal[True] = True,
        cast_mask: bool = ...,
    ) -> Tuple[Tensor, Tensor]: ...

    @overload
    def generate(  # https://github.com/pylint-dev/pylint/issues/5264 - pylint: disable=signature-differs
        self,
        generate_missing_mask: bool,
        cast_mask: bool = ...,
    ) -> Tuple[Tensor, Optional[Tensor]]: ...

    def generate(
        self,
        generate_missing_mask: bool = True,
        cast_mask: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return (
            self.generate_values(),
            self.generate_mask(generate_missing_mask=generate_missing_mask, cast_mask=cast_mask),
        )

    @overload
    def generate_mask(
        self,
        generate_missing_mask: Literal[True] = ...,
        cast_mask: bool = ...,
    ) -> Tensor: ...

    @overload
    def generate_mask(  # https://github.com/pylint-dev/pylint/issues/5264 - pylint: disable=signature-differs
        self,
        generate_missing_mask: Union[bool, Literal[False]],
        cast_mask: bool = ...,
    ) -> Optional[Tensor]: ...

    def generate_mask(
        self,
        generate_missing_mask: bool = True,
        cast_mask: bool = False,
    ) -> Optional[Tensor]:
        target_dtype = self._values.dtype if cast_mask else torch_bool
        if self._mask is not None:
            return self._mask.to(target_dtype)
        return (
            ones(
                reduce_channel_shape_to_ones(self._values.shape, self._n_channel_dims),
                device=self._values.device,
                dtype=target_dtype,
            )
            if generate_missing_mask
            else None
        )

    def generate_values(
        self,
    ) -> Tensor:
        return self._affine_transformation(self._values)

    def _get_tensors(self) -> Mapping[str, Tensor]:
        tensors = {"values": self._values}
        if self._mask is not None:
            tensors["mask"] = self._mask
        return tensors

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"affine_transformation": self._affine_transformation}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "MaskedTensor":
        if not isinstance(children["affine_transformation"], IAffineTransformation):
            raise ValueError("Invalid affine transformation object for masked tensor.")
        return MaskedTensor(
            values=tensors["values"],
            mask=tensors.get("mask"),
            n_channel_dims=self._n_channel_dims,
            affine_transformation=children["affine_transformation"],
        )

    def apply_affine(self, affine_transformation: IAffineTransformation) -> "MaskedTensor":
        return MaskedTensor(
            values=self._values,
            mask=self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=affine_transformation.compose(self._affine_transformation),
        )

    def has_mask(self) -> bool:
        return self._mask is not None

    def clear_mask(self) -> "IMaskedTensor":
        return self.modify_mask(None)

    def as_slice(self, target_shape: Sequence[int]) -> None:
        return None

    def reduce(self) -> "MaskedTensor":
        return MaskedTensor(
            values=self.generate_values(),
            mask=self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=IdentityAffineTransformation(
                n_dims=self.channels_shape[0], dtype=self._values.dtype, device=self._values.device
            ),
        )

    def modify_values(self, values: Tensor) -> "MaskedTensor":
        return MaskedTensor(
            values=values,
            mask=self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=None,
        )

    def modify_mask(self, mask: Optional[Tensor]) -> "MaskedTensor":
        return MaskedTensor(
            values=self._values,
            mask=mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=self._affine_transformation,
        )

    def __repr__(self) -> str:
        return (
            f"MaskedTensor(values={self._values}, mask={self._mask}, "
            f"n_channel_dims={self._n_channel_dims}, "
            f"affine_transformation={self._affine_transformation})"
        )


class VoxelCoordinateGrid(IMaskedTensor, BaseTensorLikeWrapper):
    """Voxel coordinate grid"""

    def __init__(
        self,
        shape: Sequence[int],
        affine_transformation: Optional[IAffineTransformation] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
        reduce_to_slice_tol: float = 1e-5,
    ) -> None:
        self._spatial_shape = shape
        self._channels_shape = (len(shape),)
        self._shape = (1,) + self._channels_shape + tuple(self._spatial_shape)
        self._dtype = get_default_dtype() if dtype is None else dtype
        self._device = torch_device("cpu") if device is None else device
        self._affine_transformation: IAffineTransformation = (
            IdentityAffineTransformation(
                n_dims=len(self._spatial_shape), dtype=self._dtype, device=self._device
            )
            if affine_transformation is None
            else affine_transformation
        )
        self._reduce_to_slice_tol = reduce_to_slice_tol

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    @property
    def channels_shape(self) -> Sequence[int]:
        return self._channels_shape

    @property
    def spatial_shape(self) -> Sequence[int]:
        return self._spatial_shape

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"affine_transformation": self._affine_transformation}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "VoxelCoordinateGrid":
        if not isinstance(children["affine_transformation"], IAffineTransformation):
            raise ValueError("Invalid affine transformation object for voxel grid.")
        return VoxelCoordinateGrid(
            shape=self._spatial_shape,
            affine_transformation=children["affine_transformation"],
            dtype=children["affine_transformation"].dtype,
            device=children["affine_transformation"].device,
        )

    @overload
    def generate(
        self,
        generate_missing_mask: Literal[True] = True,
        cast_mask: bool = ...,
    ) -> Tuple[Tensor, Tensor]: ...

    @overload
    def generate(  # https://github.com/pylint-dev/pylint/issues/5264 - pylint: disable=signature-differs
        self,
        generate_missing_mask: bool,
        cast_mask: bool = ...,
    ) -> Tuple[Tensor, Optional[Tensor]]: ...

    def generate(
        self,
        generate_missing_mask: bool = True,
        cast_mask: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return (
            self.generate_values(),
            self.generate_mask(generate_missing_mask=generate_missing_mask, cast_mask=cast_mask),
        )

    @overload
    def generate_mask(
        self,
        generate_missing_mask: Literal[True] = ...,
        cast_mask: bool = ...,
    ) -> Tensor: ...

    @overload
    def generate_mask(  # https://github.com/pylint-dev/pylint/issues/5264 - pylint: disable=signature-differs
        self,
        generate_missing_mask: Union[bool, Literal[False]],
        cast_mask: bool = ...,
    ) -> Optional[Tensor]: ...

    def generate_mask(
        self,
        generate_missing_mask: bool = True,
        cast_mask: bool = False,
    ) -> Optional[Tensor]:
        target_dtype = self._dtype if cast_mask else torch_bool
        return (
            ones(
                (1, 1) + tuple(self._spatial_shape),
                device=self._device,
                dtype=target_dtype,
            )
            if generate_missing_mask
            else None
        )

    def generate_values(
        self,
    ) -> Tensor:
        return self._affine_transformation(
            generate_voxel_coordinate_grid(
                self._spatial_shape, device=self._device, dtype=self._dtype
            )
        )

    def apply_affine(self, affine_transformation: IAffineTransformation) -> "VoxelCoordinateGrid":
        return VoxelCoordinateGrid(
            shape=self._spatial_shape,
            affine_transformation=affine_transformation.compose(self._affine_transformation),
            dtype=self._dtype,
            device=self._device,
        )

    def has_mask(self) -> bool:
        return False

    def clear_mask(self) -> "VoxelCoordinateGrid":
        return self

    def detach(self) -> "VoxelCoordinateGrid":
        return self

    def modify_values(self, values: Tensor) -> IMaskedTensor:
        return MaskedTensor(
            values=values,
            mask=None,
            n_channel_dims=1,
            affine_transformation=None,
        )

    def modify_mask(self, mask: Optional[Tensor]) -> IMaskedTensor:
        if mask is None:
            return self
        return MaskedTensor(
            values=self.generate_values(),
            mask=mask,
            n_channel_dims=1,
            affine_transformation=None,
        )

    def as_slice(
        self, target_shape: Sequence[int]
    ) -> Optional[Tuple[Union["ellipsis", slice], ...]]:
        """Reduce the grid to slice on target shape, if possible"""
        transformation_matrix = self._affine_transformation.as_cpu_matrix()
        if transformation_matrix is None:
            return None
        other_than_channel_dims = get_other_than_channel_dims(transformation_matrix.ndim, 2)
        for dim in other_than_channel_dims:
            transformation_matrix = transformation_matrix.squeeze(dim)
        if transformation_matrix.ndim != 2:
            return None
        scale = diagonal(transformation_matrix[:-1, :-1])
        if not allclose(
            diag(scale), transformation_matrix[:-1, :-1], atol=self._reduce_to_slice_tol
        ):
            return None
        translation = transformation_matrix[:-1, -1]
        if (
            torch_any(translation < -self._reduce_to_slice_tol)
            or not allclose(translation.round(), translation, atol=self._reduce_to_slice_tol)
            or not allclose(scale.round(), scale, atol=self._reduce_to_slice_tol)
        ):
            return None
        scale = torch_round(scale).type(torch_int32)
        translation = torch_round(translation).type(torch_int32)
        target_shape_tensor = tensor(target_shape, dtype=torch_int32)
        shape_tensor = tensor(self.spatial_shape, dtype=torch_int32)
        slice_ends = (shape_tensor - 1) * scale + translation + 1
        if torch_any(slice_ends > target_shape_tensor):
            return None
        return (...,) + tuple(
            slice(int(slice_start), int(slice_end), int(step_size))
            for (slice_start, slice_end, step_size) in zip(translation, slice_ends, scale)
        )

    def reduce(self) -> IMaskedTensor:
        return MaskedTensor(
            values=self.generate_values(),
            mask=None,
            n_channel_dims=1,
            affine_transformation=IdentityAffineTransformation(
                n_dims=len(self._shape), dtype=self._dtype, device=self._device
            ),
        )

    def __repr__(self) -> str:
        return (
            f"VoxelCoordinateGrid(shape={self._spatial_shape}, "
            f"affine_transformation={self._affine_transformation}, "
            f"dtype={self._dtype}, device={self._device}, "
            f"reduce_to_slice_tol={self._reduce_to_slice_tol})"
        )
