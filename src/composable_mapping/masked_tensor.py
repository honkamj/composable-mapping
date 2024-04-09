"""Masked tensors"""

from typing import Mapping, Optional, Sequence, Tuple, Union

from torch import Tensor, allclose
from torch import any as torch_any
from torch import cat
from torch import device as torch_device
from torch import diag, diagonal
from torch import dtype as torch_dtype
from torch import get_default_dtype
from torch import int32 as torch_int32
from torch import ones
from torch import round as torch_round
from torch import tensor

from composable_mapping.base import BaseTensorLikeWrapper

from .affine import IdentityAffineTransformation
from .dense_deformation import generate_voxel_coordinate_grid
from .interface import IAffineTransformation, IMaskedTensor, ITensorLike
from .util import index_by_channel_dims, reduce_channel_shape_to_ones


def concatenate_channels(
    *masked_tensors: IMaskedTensor, combine_mask: bool = True, channel_index: int = 0
) -> IMaskedTensor:
    """Concatenate masked tensors along the channel dimension

    Args:
        masked_tensors: Masked tensors to concatenate
        combine_mask: Combine masks of the masked tensors by multiplying,
            otherwise the mask of the first tensor is used.
        channel_index: Index of the channel dimension starting from the first
            channel dimension (second dimension of the tensor)
    """
    if not all(
        len(masked_tensor.channels_shape) == len(masked_tensors[0].channels_shape)
        for masked_tensor in masked_tensors
    ):
        raise ValueError("Lengths of channel shapes of masked tensors must be the same")
    if channel_index >= len(masked_tensors[0].channels_shape) or channel_index < 0:
        raise ValueError("Invalid channel index")
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
    if combine_mask:
        mask = masked_tensors[0].generate_mask(generate_missing_mask=False)
        for masked_tensor in masked_tensors[1:]:
            update_mask = masked_tensor.generate_mask(generate_missing_mask=False)
            if mask is not None and update_mask is not None:
                mask = mask * update_mask
    else:
        mask = masked_tensors[0].generate_mask(generate_missing_mask=False)
    return MaskedTensor(
        values=values,
        mask=mask,
        n_channel_dims=n_channel_dims,
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
        return self._values.shape[1 + self._n_channel_dims :]

    @property
    def channels_shape(self) -> Sequence[int]:
        first_spatial_dim = index_by_channel_dims(self._values.ndim, -1, self._n_channel_dims) + 1
        return self._values.shape[first_spatial_dim:]

    def generate(
        self,
        generate_missing_mask: bool = True,
    ):
        return (
            self.generate_values(),
            self.generate_mask(generate_missing_mask=generate_missing_mask),
        )

    def generate_mask(
        self,
        generate_missing_mask: bool = True,
    ):
        if self._mask is not None:
            return self._mask
        return (
            ones(
                reduce_channel_shape_to_ones(self._values.shape, self._n_channel_dims),
                device=self._values.device,
                dtype=self._values.dtype,
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
        return MaskedTensor(
            values=self._values,
            mask=None,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=self._affine_transformation,
        )

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
                n_dims=len(self._shape), dtype=self._dtype, device=self._device
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

    def generate(self, generate_missing_mask: bool = True):
        return (
            self.generate_values(),
            self.generate_mask(generate_missing_mask=generate_missing_mask),
        )

    def generate_mask(self, generate_missing_mask: bool = True):
        return (
            ones(
                (1, 1) + tuple(self._spatial_shape),
                device=self._device,
                dtype=self._dtype,
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

    def as_slice(
        self, target_shape: Sequence[int]
    ) -> Optional[Tuple[Union["ellipsis", slice], ...]]:
        """Reduce the grid to slice on target shape, if possible"""
        transformation_matrix = self._affine_transformation.as_cpu_matrix()
        if transformation_matrix is None:
            return None
        transformation_matrix = transformation_matrix.squeeze()
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
        shape_tensor = tensor(self.shape, dtype=torch_int32)
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
