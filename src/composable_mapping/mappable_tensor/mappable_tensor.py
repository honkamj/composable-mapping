"""Masked tensors"""

from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple, Union, overload

from torch import Tensor, allclose
from torch import any as torch_any
from torch import bool as torch_bool
from torch import device as torch_device
from torch import diag, diagonal
from torch import dtype as torch_dtype
from torch import get_default_dtype
from torch import int32 as torch_int32
from torch import ones
from torch import round as torch_round
from torch import tensor, zeros

from composable_mapping.dense_deformation import generate_voxel_coordinate_grid
from composable_mapping.tensor_like import BaseTensorLikeWrapper, ITensorLike
from composable_mapping.util import (
    broadcast_shapes_in_parts_splitted,
    broadcast_shapes_in_parts_to_single_shape,
    broadcast_tensors_in_parts,
    broadcast_to_in_parts,
    combine_optional_masks,
    get_batch_dims,
    get_channels_shape,
    get_spatial_dims,
    get_spatial_shape,
    optional_add,
    reduce_channel_shape_to_ones,
    split_shape,
)

from .affine_transformation import IAffineTransformation, IdentityAffineTransformation

REDUCE_TO_SLICE_TOLERANCE = 1e-5


class MappableTensor(BaseTensorLikeWrapper):
    """Abstract base class for masked tensors

    Defined as abstract since the constructor is unsafe to use directly.
    """

    def __init__(
        self,
        spatial_shape: Sequence[int],
        displacements: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        n_channel_dims: int = 1,
        affine_transformation_on_displacements: Optional[IAffineTransformation] = None,
        affine_transformation_on_voxel_grid: Optional[IAffineTransformation] = None,
    ) -> None:
        self._spatial_shape = tuple(spatial_shape)
        self._displacements = displacements
        self._mask = mask
        self._n_channel_dims = n_channel_dims
        if displacements is None and affine_transformation_on_displacements is not None:
            raise ValueError("Affine transformation on displacements requires displacements")
        if displacements is not None and affine_transformation_on_displacements is None:
            self._affine_transformation_on_displacements: Optional[IAffineTransformation] = (
                IdentityAffineTransformation(
                    get_channels_shape(displacements.shape, n_channel_dims=n_channel_dims)[-1],
                    dtype=displacements.dtype,
                    device=displacements.device,
                )
            )
        else:
            self._affine_transformation_on_displacements = affine_transformation_on_displacements
        self._affine_transformation_on_voxel_grid: Optional[IAffineTransformation] = (
            affine_transformation_on_voxel_grid
        )

    def _get_tensors(self) -> Mapping[str, Tensor]:
        tensors: Dict[str, Tensor] = {}
        if self._displacements is not None:
            tensors["displacements"] = self._displacements
        if self._mask is not None:
            tensors["mask"] = self._mask
        return tensors

    def _get_children(self) -> Mapping[str, ITensorLike]:
        children: Dict[str, ITensorLike] = {}
        if self._affine_transformation_on_displacements is not None:
            children["affine_transformation_on_displacements"] = (
                self._affine_transformation_on_displacements
            )
        if self._affine_transformation_on_voxel_grid is not None:
            children["affine_transformation_on_voxel_grid"] = (
                self._affine_transformation_on_voxel_grid
            )
        return children

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "MappableTensor":
        if "affine_transformation_on_displacements" in children:
            if not isinstance(
                children["affine_transformation_on_displacements"], IAffineTransformation
            ):
                raise ValueError("Invalid children for mappable tensor")
            affine_transformation_on_displacements: Optional[IAffineTransformation] = children[
                "affine_transformation_on_displacements"
            ]

        else:
            affine_transformation_on_displacements = None
        if "affine_transformation_on_voxel_grid" in children:
            if not isinstance(
                children["affine_transformation_on_voxel_grid"], IAffineTransformation
            ):
                raise ValueError("Invalid children for mappable tensor")
            affine_transformation_on_voxel_grid: Optional[IAffineTransformation] = children[
                "affine_transformation_on_voxel_grid"
            ]
        else:
            affine_transformation_on_voxel_grid = None
        return MappableTensor(
            spatial_shape=self._spatial_shape,
            displacements=tensors.get("displacements", self._displacements),
            mask=tensors.get("mask", self._mask),
            n_channel_dims=self._n_channel_dims,
            affine_transformation_on_displacements=affine_transformation_on_displacements,
            affine_transformation_on_voxel_grid=affine_transformation_on_voxel_grid,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the tensor"""
        if self._affine_transformation_on_voxel_grid is None:
            voxel_grid_shape: Optional[Tuple[int, ...]] = None
        else:
            voxel_grid_shape = (1, len(self.spatial_shape)) + tuple(self._spatial_shape)
            voxel_grid_shape = self._affine_transformation_on_voxel_grid.get_output_shape(
                (1, len(self.spatial_shape)) + tuple(self._spatial_shape)
            )
        if self._displacements is None:
            values_shape: Optional[Tuple[int, ...]] = None
        else:
            values_shape = self._displacements.shape
            if self._affine_transformation_on_displacements is not None:
                values_shape = self._affine_transformation_on_displacements.get_output_shape(
                    values_shape, n_channel_dims=self._n_channel_dims
                )
        if voxel_grid_shape is not None and values_shape is not None:
            return broadcast_shapes_in_parts_to_single_shape(
                values_shape, voxel_grid_shape, n_channel_dims=(self._n_channel_dims, 1)
            )
        if values_shape is not None:
            return values_shape
        if voxel_grid_shape is not None:
            return voxel_grid_shape
        raise ValueError("Either displacements or affine transformation on voxel grid must be set")

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        """Shape of the spatial dimensions"""
        return self._spatial_shape

    @property
    def channels_shape(self) -> Tuple[int, ...]:
        """Shape of the channel dimensions"""
        return get_channels_shape(self.shape, self._n_channel_dims)

    @property
    def n_channel_dims(self) -> int:
        """Number of channel dimensions"""
        return self._n_channel_dims

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
        """Generate values and mask contained by the mappable tensor

        Args:
            generate_mask: Generate mask of ones if the tensor does not contain an explicit mask
            cast_mask: Mask is stored as a boolean tensor, cast it to dtype of values if True
        """
        return self.generate_values(), self.generate_mask(
            generate_missing_mask=generate_missing_mask, cast_mask=cast_mask
        )

    def generate_values(
        self,
    ) -> Tensor:
        """Generate values contained by the mappable tensor"""
        batch_shape, channels_shape, spatial_shape = split_shape(
            self.shape, n_channel_dims=self._n_channel_dims
        )
        displacements = self._displacements
        if displacements is not None:
            if self._affine_transformation_on_displacements is not None:
                displacements = self._affine_transformation_on_displacements(
                    displacements, n_channel_dims=self._n_channel_dims
                )
            displacements = broadcast_to_in_parts(
                displacements,
                batch_shape=batch_shape,
                channels_shape=channels_shape,
                spatial_shape=spatial_shape,
                n_channel_dims=self._n_channel_dims,
            )
        if (
            self._affine_transformation_on_voxel_grid is None
            or self._affine_transformation_on_voxel_grid.is_zero()
        ):
            grid: Optional[Tensor] = None
        else:
            voxel_grid = generate_voxel_coordinate_grid(
                self.spatial_shape, device=self.device, dtype=self.dtype
            )
            grid = self._affine_transformation_on_voxel_grid(voxel_grid, n_channel_dims=1)
            grid = broadcast_to_in_parts(
                grid,
                batch_shape=batch_shape,
                channels_shape=channels_shape,
                spatial_shape=spatial_shape,
                n_channel_dims=1,
            )
        values = optional_add(displacements, grid)
        if values is None:
            return zeros(self.shape, dtype=self.dtype, device=self.device)
        return values

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
        """Generate mask contained by the mappable tensor

        Args:
            generate_mask: Generate mask of ones if the tensor does not contain an explicit mask
            cast_mask: Mask is stored as a boolean tensor, cast it to dtype of values if True
        """
        target_dtype = self.dtype if cast_mask else torch_bool
        if self._mask is not None:
            return self._mask.to(target_dtype)
        return (
            ones(
                reduce_channel_shape_to_ones(self.shape, self._n_channel_dims),
                device=self.device,
                dtype=target_dtype,
            )
            if generate_missing_mask
            else None
        )

    @property
    def displacements(self) -> Optional[Tensor]:
        """Displacements contained by the mappable tensor

        These might not be broadcasted to the full shape of the tensor
        """
        return self._displacements

    @property
    def affine_transformation_on_displacements(self) -> Optional[IAffineTransformation]:
        """Affine transformation on displacements, if available"""
        return self._affine_transformation_on_displacements

    @property
    def affine_transformation_on_voxel_grid(self) -> Optional[IAffineTransformation]:
        """Affine transformation on voxel grid, if available"""
        return self._affine_transformation_on_voxel_grid

    def transform(self, affine_transformation: IAffineTransformation) -> "MappableTensor":
        """Apply affine mapping to the last channel dimension of the tensor

        The affine transformation is not evaluated immidiately, only the composition is stored.
        """
        if self._displacements is None:
            affine_transformation_on_displacements = None
        elif self._affine_transformation_on_displacements is None:
            affine_transformation_on_displacements = affine_transformation
        else:
            affine_transformation_on_displacements = (
                affine_transformation @ self._affine_transformation_on_displacements
            )
        if self._affine_transformation_on_voxel_grid is None:
            affine_transformation_on_voxel_grid = None
        else:
            affine_transformation_on_voxel_grid = (
                affine_transformation @ self._affine_transformation_on_voxel_grid
            )
        return MappableTensor(
            spatial_shape=self._spatial_shape,
            displacements=(None if self._displacements is None else self._displacements),
            mask=None if self._mask is None else self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation_on_displacements=affine_transformation_on_displacements,
            affine_transformation_on_voxel_grid=affine_transformation_on_voxel_grid,
        )

    def __add__(self, other: "MappableTensor") -> "MappableTensor":
        """Sum two mappable tensors"""
        self_mask = self.generate_mask(generate_missing_mask=False, cast_mask=False)
        other_mask = other.generate_mask(generate_missing_mask=False, cast_mask=False)
        if self_mask is not None and other_mask is not None:
            self_mask, other_mask = broadcast_tensors_in_parts(
                self_mask,
                other_mask,
                n_channel_dims=(self.n_channel_dims, other.n_channel_dims),
            )
        mask = combine_optional_masks([self_mask, other_mask])

        self_displacements = self._displacements
        other_displacements = other.displacements

        if self_displacements is None:
            displacements = other_displacements
            affine_transformation_on_displacements = other.affine_transformation_on_displacements
            output_spatial_shape = other.spatial_shape
            n_channel_dims = other.n_channel_dims
        elif other_displacements is None:
            displacements = self_displacements
            affine_transformation_on_displacements = self._affine_transformation_on_displacements
            output_spatial_shape = self.spatial_shape
            n_channel_dims = self.n_channel_dims
        else:
            affine_transformation_on_displacements = None
            if self.spatial_shape == other.spatial_shape:
                transformed_self_displacements = (
                    self_displacements
                    if self._affine_transformation_on_displacements is None
                    else self._affine_transformation_on_displacements(
                        self_displacements, n_channel_dims=self.n_channel_dims
                    )
                )
                transformed_other_displacements = (
                    other_displacements
                    if other.affine_transformation_on_displacements is None
                    else other.affine_transformation_on_displacements(other_displacements)
                )
            else:
                transformed_self_displacements = self.generate_values()
                transformed_other_displacements = other.generate_values()
            batch_shape, channels_shape, spatial_shape = broadcast_shapes_in_parts_splitted(
                transformed_self_displacements.shape,
                transformed_other_displacements.shape,
                n_channel_dims=(self.n_channel_dims, other.n_channel_dims),
            )
            assert channels_shape is not None and spatial_shape is not None
            output_spatial_shape = spatial_shape
            n_channel_dims = len(channels_shape)
            transformed_self_displacements = broadcast_to_in_parts(
                transformed_self_displacements,
                batch_shape=batch_shape,
                channels_shape=channels_shape,
                spatial_shape=spatial_shape,
            )
            transformed_other_displacements = broadcast_to_in_parts(
                transformed_other_displacements,
                batch_shape=batch_shape,
                channels_shape=channels_shape,
                spatial_shape=spatial_shape,
            )
            displacements = transformed_self_displacements + transformed_other_displacements

        if self.spatial_shape == other.spatial_shape:
            affine_transformation_on_voxel_grid = optional_add(
                self.affine_transformation_on_voxel_grid,
                other.affine_transformation_on_voxel_grid,
            )
        else:
            affine_transformation_on_voxel_grid = None

        return MappableTensor(
            spatial_shape=output_spatial_shape,
            displacements=displacements,
            mask=mask,
            n_channel_dims=n_channel_dims,
            affine_transformation_on_voxel_grid=affine_transformation_on_voxel_grid,
            affine_transformation_on_displacements=affine_transformation_on_displacements,
        )

    def __sub__(self, other: "MappableTensor") -> "MappableTensor":
        """Substract two mappable tensors"""
        return self.__add__(-other)

    def __neg__(self) -> "MappableTensor":
        """Negate the mappable tensor"""
        return MappableTensor(
            spatial_shape=self._spatial_shape,
            displacements=self._displacements,
            mask=None if self._mask is None else self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation_on_displacements=(
                None
                if self._affine_transformation_on_displacements is None
                else -self._affine_transformation_on_displacements
            ),
            affine_transformation_on_voxel_grid=(
                None
                if self._affine_transformation_on_voxel_grid is None
                else -self._affine_transformation_on_voxel_grid
            ),
        )

    def has_mask(self) -> bool:
        """Returns whether the tensor has an explicit mask"""
        return self._mask is not None

    def clear_mask(self) -> "MappableTensor":
        """Remove mask from the tensor"""
        return self.modify_mask(None)

    def as_slice(
        self, target_shape: Sequence[int]
    ) -> Optional[Tuple[Union["ellipsis", slice], ...]]:
        """Reduce the grid to slice on target shape, if possible"""
        if self._displacements is not None or self._affine_transformation_on_voxel_grid is None:
            return None
        transformation_matrix = self._affine_transformation_on_voxel_grid.as_host_matrix()
        if transformation_matrix is None:
            return None
        transformation_matrix = transformation_matrix.squeeze(
            dim=get_batch_dims(transformation_matrix.ndim, 2)
            + get_spatial_dims(transformation_matrix.ndim, 2)
        )
        if transformation_matrix.ndim != 2:
            return None
        scale = diagonal(transformation_matrix[:-1, :-1])
        if not allclose(
            diag(scale), transformation_matrix[:-1, :-1], atol=REDUCE_TO_SLICE_TOLERANCE
        ):
            return None
        translation = transformation_matrix[:-1, -1]
        if (
            torch_any(translation < -REDUCE_TO_SLICE_TOLERANCE)
            or (not allclose(translation.round(), translation, atol=REDUCE_TO_SLICE_TOLERANCE))
            or not allclose(scale.round(), scale, atol=REDUCE_TO_SLICE_TOLERANCE)
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

    def reduce(self) -> "PlainTensor":
        """Reduce the masked tensor to a plain tensor"""
        values, mask = self.generate(generate_missing_mask=False, cast_mask=False)
        return PlainTensor(
            values=values,
            mask=mask,
            n_channel_dims=self._n_channel_dims,
        )

    def modify_values(self, values: Tensor) -> "MappableTensor":
        """Modify values of the tensor"""
        return MappableTensor(
            spatial_shape=self._spatial_shape,
            displacements=values,
            mask=self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation_on_displacements=None,
            affine_transformation_on_voxel_grid=None,
        )

    def mask_and(self, mask: Optional[Tensor]) -> "MappableTensor":
        """Combine mask with logical and"""
        return self.modify_mask(
            combine_optional_masks([self._mask, mask], n_channel_dims=self._n_channel_dims)
        )

    def modify_mask(self, mask: Optional[Tensor]) -> "MappableTensor":
        """Modify mask of the tensor"""
        return MappableTensor(
            spatial_shape=self._spatial_shape,
            displacements=self._displacements,
            mask=mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation_on_displacements=self._affine_transformation_on_displacements,
            affine_transformation_on_voxel_grid=self._affine_transformation_on_voxel_grid,
        )

    def modify(self, values: Tensor, mask: Optional[Tensor]) -> "MappableTensor":
        """Modify values and mask of the tensor"""
        return MappableTensor(
            spatial_shape=self._spatial_shape,
            displacements=values,
            mask=mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation_on_displacements=None,
            affine_transformation_on_voxel_grid=None,
        )

    def __repr__(self) -> str:
        return (
            f"MappableTensor(spatial_shape={self._spatial_shape}, "
            f"dtype={self.dtype}, device={self.device}, "
            f"displacements={self._displacements}, mask={self._mask}, "
            f"n_channel_dims={self._n_channel_dims}, "
            f"affine_transformation_on_displacements="
            f"{self._affine_transformation_on_displacements}, "
            f"affine_transformation_on_voxel_grid={self._affine_transformation_on_voxel_grid})"
        )


class PlainTensor(MappableTensor):
    """Plain tensor

    Arguments:
        values: Tensor with shape (batch_size, *channel_dims, *spatial_dims)
        mask: Tensor with shape (batch_size, (1,) * n_channel_dims, *spatial_dims)
        n_channel_dims: Number of channel dimensions
    """

    def __init__(
        self,
        values: Tensor,
        mask: Optional[Tensor] = None,
        n_channel_dims: int = 1,
    ) -> None:
        super().__init__(
            spatial_shape=get_spatial_shape(values.shape, n_channel_dims),
            displacements=values,
            mask=mask,
            n_channel_dims=n_channel_dims,
        )


class VoxelGrid(MappableTensor):
    """Voxel grid with optional mask

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
    ) -> None:
        dtype = get_default_dtype() if dtype is None else dtype
        device = torch_device("cpu") if device is None else device
        super().__init__(
            spatial_shape=spatial_shape,
            mask=mask,
            n_channel_dims=1,
            affine_transformation_on_voxel_grid=IdentityAffineTransformation(
                len(spatial_shape), dtype=dtype, device=device
            ),
        )
