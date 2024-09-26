"""Masked tensors"""

from typing import Literal, Optional, Sequence, Tuple, Union, overload

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
from torch import stack, tensor, zeros

from .affine import (
    IdentityAffineTransformation,
    ZeroAffineTransformation,
    get_coordinates_affine_dimensionality,
)
from .dense_deformation import generate_voxel_coordinate_grid
from .interface import IAffineTransformation, IMaskedTensor
from .util import (
    broadcast_tensors_around_channel_dims,
    broadcast_to_shape_around_channel_dims,
    combine_optional_masks,
    get_other_than_channel_dims,
    index_by_channel_dims,
    optional_add,
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


class _MaskedTensor(IMaskedTensor):
    def __init__(
        self,
        shape: Sequence[int],
        dtype: torch_dtype,
        device: torch_device,
        values: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        n_channel_dims: int = 1,
        affine_transformation: Optional[IAffineTransformation] = None,
        affine_transformation_on_voxel_grid: Optional[IAffineTransformation] = None,
        reduce_to_slice_tol: float = 1e-5,
    ) -> None:
        self._shape = shape
        self._dtype = dtype
        self._device = device
        self._values = values
        self._mask = mask
        self._n_channel_dims = n_channel_dims
        self._affine_transformation: IAffineTransformation = (
            IdentityAffineTransformation(
                get_coordinates_affine_dimensionality(self.channels_shape),
                dtype=self._dtype,
                device=self._device,
            )
            if affine_transformation is None
            else affine_transformation
        )
        self._affine_transformation_on_voxel_grid: IAffineTransformation = (
            ZeroAffineTransformation(
                n_input_dims=len(self.spatial_shape),
                n_output_dims=self.channels_shape[-1],
                dtype=self._dtype,
                device=self._device,
            )
            if affine_transformation_on_voxel_grid is None
            else affine_transformation_on_voxel_grid
        )
        self._reduce_to_slice_tol = reduce_to_slice_tol

    @property
    def shape(self) -> Sequence[int]:
        return self._shape

    @property
    def spatial_shape(self) -> Sequence[int]:
        first_spatial_dim = index_by_channel_dims(len(self._shape), -1, self._n_channel_dims) + 1
        return self._shape[first_spatial_dim:]

    @property
    def channels_shape(self) -> Sequence[int]:
        first_channel_dim = index_by_channel_dims(len(self._shape), 0, self._n_channel_dims)
        return self._shape[first_channel_dim : first_channel_dim + self._n_channel_dims]

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
        return self.generate_values(), self.generate_mask(
            generate_missing_mask=generate_missing_mask, cast_mask=cast_mask
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
        if self._mask is not None:
            return self._mask.to(target_dtype)
        return (
            ones(
                reduce_channel_shape_to_ones(self._shape, self._n_channel_dims),
                device=self._device,
                dtype=target_dtype,
            )
            if generate_missing_mask
            else None
        )

    def generate_values(
        self,
    ) -> Tensor:
        if self._values is None:
            displacements: Optional[Tensor] = None
        else:
            displacements = self._affine_transformation(
                self._values, n_channel_dims=self._n_channel_dims
            )
            displacements = broadcast_to_shape_around_channel_dims(
                displacements, self._shape, n_channel_dims=self._n_channel_dims
            )
        if self._affine_transformation_on_voxel_grid.is_zero(
            n_input_dims=len(self.spatial_shape), n_output_dims=self.channels_shape[-1]
        ):
            grid: Optional[Tensor] = None
        else:
            voxel_grid = generate_voxel_coordinate_grid(
                self.spatial_shape, device=self._device, dtype=self._dtype
            )
            grid = self._affine_transformation_on_voxel_grid(voxel_grid, n_channel_dims=1)
            if self._n_channel_dims > 1:
                grid = grid[(slice(None),) + (None,) * (self._n_channel_dims - 1)]
            grid = broadcast_to_shape_around_channel_dims(
                grid, self._shape, n_channel_dims=self._n_channel_dims
            )
        values = optional_add(displacements, grid)
        if values is None:
            return zeros(self._shape, dtype=self._dtype, device=self._device)
        return values

    @property
    def dtype(
        self,
    ) -> torch_dtype:
        return self._dtype

    @property
    def device(
        self,
    ) -> torch_device:
        return self._device

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "_MaskedTensor":
        return _MaskedTensor(
            shape=self._shape,
            dtype=self._dtype if dtype is None else dtype,
            device=self._device if device is None else device,
            values=None if self._values is None else self._values.to(dtype=dtype, device=device),
            mask=None if self._mask is None else self._mask.to(dtype=dtype, device=device),
            n_channel_dims=self._n_channel_dims,
            affine_transformation=self._affine_transformation.cast(dtype=dtype, device=device),
            affine_transformation_on_voxel_grid=self._affine_transformation_on_voxel_grid.cast(
                dtype=dtype, device=device
            ),
        )

    def detach(self) -> "_MaskedTensor":
        return _MaskedTensor(
            shape=self._shape,
            dtype=self._dtype,
            device=self._device,
            values=None if self._values is None else self._values.detach(),
            mask=None if self._mask is None else self._mask.detach(),
            n_channel_dims=self._n_channel_dims,
            affine_transformation=self._affine_transformation.detach(),
            affine_transformation_on_voxel_grid=self._affine_transformation_on_voxel_grid.detach(),
        )

    def apply_affine(self, affine_transformation: IAffineTransformation) -> IMaskedTensor:
        return _MaskedTensor(
            shape=self._shape,
            dtype=self._dtype,
            device=self._device,
            values=self._values,
            mask=self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=affine_transformation @ self._affine_transformation,
            affine_transformation_on_voxel_grid=affine_transformation
            @ self._affine_transformation_on_voxel_grid,
        )

    def add_grid(self, affine_transformation_on_voxel_grid: IAffineTransformation) -> IMaskedTensor:
        return _MaskedTensor(
            shape=self._shape,
            dtype=self._dtype,
            device=self._device,
            values=self._values,
            mask=self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=self._affine_transformation,
            affine_transformation_on_voxel_grid=affine_transformation_on_voxel_grid
            + self._affine_transformation_on_voxel_grid,
        )

    def has_mask(self) -> bool:
        return self._mask is not None

    def clear_mask(self) -> IMaskedTensor:
        return self.modify_mask(None)

    def as_slice(
        self, target_shape: Sequence[int]
    ) -> Optional[Tuple[Union["ellipsis", slice], ...]]:
        """Reduce the grid to slice on target shape, if possible"""
        if self._values is not None:
            return None
        transformation_matrix = self._affine_transformation_on_voxel_grid.as_cpu_matrix()
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
        if not allclose(
            translation.round(), translation, atol=self._reduce_to_slice_tol
        ) or not allclose(scale.round(), scale, atol=self._reduce_to_slice_tol):
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
        values, mask = self.generate(generate_missing_mask=False, cast_mask=False)
        return _MaskedTensor(
            shape=self._shape,
            dtype=self._dtype,
            device=self._device,
            values=values,
            mask=mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=self._affine_transformation,
            affine_transformation_on_voxel_grid=self._affine_transformation_on_voxel_grid,
        )

    def displace(self, displacement: Tensor) -> IMaskedTensor:
        updated_values = displacement
        if self._values is not None:
            displacement, values = broadcast_tensors_around_channel_dims(
                (displacement, self._values), n_channel_dims=self._n_channel_dims
            )
            updated_values = displacement + values
        return _MaskedTensor(
            shape=self._shape,
            dtype=self._dtype,
            device=self._device,
            values=updated_values,
            mask=self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=self._affine_transformation,
            affine_transformation_on_voxel_grid=self._affine_transformation_on_voxel_grid,
        )

    def modify_values(self, values: Tensor) -> IMaskedTensor:
        return _MaskedTensor(
            shape=values.shape,
            dtype=self._dtype,
            device=self._device,
            values=values,
            mask=self._mask,
            n_channel_dims=self._n_channel_dims,
        )

    def modify_mask(self, mask: Optional[Tensor]) -> IMaskedTensor:
        return _MaskedTensor(
            shape=self._shape,
            dtype=self._dtype,
            device=self._device,
            values=self._values,
            mask=mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=self._affine_transformation,
            affine_transformation_on_voxel_grid=self._affine_transformation_on_voxel_grid,
        )

    def modify(self, values: Tensor, mask: Optional[Tensor]) -> IMaskedTensor:
        return _MaskedTensor(
            shape=values.shape,
            dtype=self._dtype,
            device=self._device,
            values=values,
            mask=mask,
            n_channel_dims=self._n_channel_dims,
        )

    def __repr__(self) -> str:
        return (
            f"_MaskedTensor(shape={self._shape}, dtype={self._dtype}, device={self._device}, "
            f"n_channel_dims={self._n_channel_dims}, "
            f"affine_transformation={self._affine_transformation}, "
            f"affine_transformation_on_voxel_grid={self._affine_transformation_on_voxel_grid})"
        )


class MaskedTensor(_MaskedTensor):
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
        affine_transformation_on_voxel_grid: Optional[IAffineTransformation] = None,
        reduce_to_slice_tol: float = 1e-5,
    ) -> None:
        super().__init__(
            shape=values.shape,
            dtype=values.dtype,
            device=values.device,
            values=values,
            mask=mask,
            n_channel_dims=n_channel_dims,
            affine_transformation=affine_transformation,
            affine_transformation_on_voxel_grid=affine_transformation_on_voxel_grid,
            reduce_to_slice_tol=reduce_to_slice_tol,
        )


class VoxelGrid(_MaskedTensor):
    """VoxelGrid with optional mask

    By default, the grid is a voxel grid with no mask

    Arguments:
        values: Tensor with shape (batch_size, *spatial_dims, n_channels)
        mask: Tensor with shape (batch_size, 1, *spatial_dims)
    """

    def __init__(
        self,
        spatial_shape: Sequence[int],
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
        mask: Optional[Tensor] = None,
        channels_shape: Optional[Sequence[int]] = None,
        batch_size: int = 1,
        reduce_to_slice_tol: float = 1e-5,
    ) -> None:
        if channels_shape is None:
            channels_shape = (len(spatial_shape),)
        shape = (batch_size,) + tuple(channels_shape) + tuple(spatial_shape)
        dtype = get_default_dtype() if dtype is None else dtype
        device = torch_device("cpu") if device is None else device
        super().__init__(
            shape=shape,
            dtype=dtype,
            device=device,
            values=None,
            mask=mask,
            n_channel_dims=len(channels_shape),
            affine_transformation_on_voxel_grid=IdentityAffineTransformation(
                n_dims=get_coordinates_affine_dimensionality(channels_shape),
                dtype=dtype,
                device=device,
            ),
            reduce_to_slice_tol=reduce_to_slice_tol,
        )
