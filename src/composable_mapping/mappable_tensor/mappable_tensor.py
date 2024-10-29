"""Masked tensors"""

from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple, Union, overload

from torch import Tensor, allclose
from torch import any as torch_any
from torch import bool as torch_bool
from torch import device as torch_device
from torch import diag, diagonal
from torch import dtype as torch_dtype
from torch import int32 as torch_int32
from torch import ones
from torch import round as torch_round
from torch import tensor, zeros

from composable_mapping.tensor_like import BaseTensorLikeWrapper, ITensorLike
from composable_mapping.util import (
    broadcast_optional_shapes_in_parts_to_single_shape,
    broadcast_tensors_in_parts,
    broadcast_to_in_parts,
    combine_optional_masks,
    get_batch_dims,
    get_batch_shape,
    get_channels_shape,
    get_spatial_dims,
    get_spatial_shape,
    optional_add,
    reduce_channels_shape_to_ones,
    split_shape,
)

from .affine_transformation import (
    HostDiagonalAffineTransformation,
    IAffineTransformation,
    IdentityAffineTransformation,
)
from .grid import GridDefinition

REDUCE_TO_SLICE_TOLERANCE = 1e-5

Number = Union[float, int]


class MappableTensor(BaseTensorLikeWrapper):
    """A tensor wrapper used as inputs for composable mappings

    It is not recommended to create instances of this class directly, but to
    use instead the constructors provided in the module, or as class methods
    of this class.
    """

    def __init__(
        self,
        displacements: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        n_channel_dims: int = 1,
        affine_transformation: Optional[IAffineTransformation] = None,
        grid: Optional[GridDefinition] = None,
    ) -> None:
        if mask is not None:
            mask_channels_shape = get_channels_shape(mask.shape, n_channel_dims)
            if mask_channels_shape != (1,) * n_channel_dims:
                raise ValueError("Mask must dimension with size 1 in all channel dimensions")
        if displacements is None and n_channel_dims != 1:
            raise ValueError("When no displacements is set, n_channel_dims must be 1")
        if displacements is None and grid is None:
            raise ValueError("Either displacements, or grid must be set.")
        self._displacements = displacements
        self._mask = mask
        self._n_channel_dims = n_channel_dims
        self._affine_transformation = (
            IdentityAffineTransformation(
                get_channels_shape(displacements.shape, n_channel_dims=n_channel_dims)[-1],
                dtype=displacements.dtype,
                device=displacements.device,
            )
            if affine_transformation is None and displacements is not None
            else affine_transformation
        )
        self._grid = grid
        self._shape = self._infer_shape()

    @classmethod
    def from_tensor(
        cls, values: Tensor, mask: Optional[Tensor] = None, n_channel_dims: int = 1
    ) -> "MappableTensor":
        """Create a mappable tensor from a tensor"""
        return MappableTensor(displacements=values, mask=mask, n_channel_dims=n_channel_dims)

    @classmethod
    def create_voxel_grid(
        cls,
        spatial_shape: Sequence[int],
        mask: Optional[Tensor] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "MappableTensor":
        """Create a voxel grid with optional mask"""
        return MappableTensor(
            mask=mask,
            n_channel_dims=1,
            grid=GridDefinition(
                spatial_shape=spatial_shape,
                affine_transformation=IdentityAffineTransformation(
                    len(spatial_shape), dtype=dtype, device=device
                ),
            ),
        )

    @property
    def _transformed_displacements_shape(self) -> Optional[Tuple[int, ...]]:
        displacements_shape: Optional[Tuple[int, ...]] = (
            None if self._displacements is None else self._displacements.shape
        )
        if self._affine_transformation is not None:
            if displacements_shape is None:
                raise ValueError("Affine transformation may be set only if displacements is set")
            displacements_shape = self._affine_transformation.get_output_shape(
                displacements_shape, n_channel_dims=self._n_channel_dims
            )
        return displacements_shape

    def _infer_shape(
        self,
    ) -> Tuple[int, ...]:
        displacements_shape = self._transformed_displacements_shape
        return broadcast_optional_shapes_in_parts_to_single_shape(
            displacements_shape,
            None if self._mask is None else self._mask.shape,
            None if self._grid is None else self._grid.shape,
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
        if self._affine_transformation is not None:
            children["affine_transformation"] = self._affine_transformation
        if self._grid is not None:
            children["grid"] = self._grid
        return children

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "MappableTensor":
        if "affine_transformation" in children:
            if not isinstance(children["affine_transformation"], IAffineTransformation):
                raise ValueError("Invalid children for mappable tensor")
            affine_transformation: Optional[IAffineTransformation] = children[
                "affine_transformation"
            ]

        else:
            affine_transformation = None
        if "grid" in children:
            if not isinstance(children["grid"], GridDefinition):
                raise ValueError("Invalid children for mappable tensor")
            grid: Optional[GridDefinition] = children["grid"]
        else:
            grid = None
        return MappableTensor(
            displacements=tensors.get("displacements", self._displacements),
            mask=tensors.get("mask", self._mask),
            n_channel_dims=self._n_channel_dims,
            affine_transformation=affine_transformation,
            grid=grid,
        )

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the tensor"""
        return self._shape

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        """Shape of the spatial dimensions"""
        return get_spatial_shape(self.shape, self._n_channel_dims)

    @property
    def channels_shape(self) -> Tuple[int, ...]:
        """Shape of the channel dimensions"""
        return get_channels_shape(self.shape, self._n_channel_dims)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Shape of the batch dimension"""
        return get_batch_shape(self.shape, self._n_channel_dims)

    @property
    def mask_shape(self) -> Tuple[int, ...]:
        """Shape of the mask"""
        return reduce_channels_shape_to_ones(self.shape, self._n_channel_dims)

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
            if self._affine_transformation is not None:
                displacements = self._affine_transformation(
                    displacements, n_channel_dims=self._n_channel_dims
                )
            displacements = broadcast_to_in_parts(
                displacements,
                batch_shape=batch_shape,
                channels_shape=channels_shape,
                spatial_shape=spatial_shape,
                n_channel_dims=self._n_channel_dims,
            )
        if self._grid is None or self._grid.is_zero():
            grid: Optional[Tensor] = None
        else:
            grid = broadcast_to_in_parts(
                self._grid.generate(),
                batch_shape=batch_shape,
                channels_shape=channels_shape,
                spatial_shape=spatial_shape,
            )
        values = optional_add(displacements, grid)
        if values is None:
            return zeros(1, dtype=self.dtype, device=self.device).expand(self.shape)
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
            batch_shape, channels_shape, spatial_shape = split_shape(
                self.mask_shape, n_channel_dims=self._n_channel_dims
            )
            return broadcast_to_in_parts(
                self._mask.to(target_dtype),
                batch_shape=batch_shape,
                channels_shape=channels_shape,
                spatial_shape=spatial_shape,
                n_channel_dims=self._n_channel_dims,
            )
        return (
            ones(
                self.mask_shape,
                device=self.device,
                dtype=target_dtype,
            )
            if generate_missing_mask
            else None
        )

    @property
    def displacements(self) -> Optional[Tensor]:
        """Displacements contained by the mappable tensor

        The displacements vector might not have the same shape as the mappable
        tensor, but is brodcastable to it. This is relatively low level, and
        should be used with caution. The recommended way to access the values
        is through the `generate_values` method.
        """
        return self._displacements

    @property
    def affine_transformation(self) -> Optional[IAffineTransformation]:
        """Affine transformation on displacements, if available

        This is relatively low level, and should be used with caution.
        """
        return self._affine_transformation

    @property
    def grid(self) -> Optional[GridDefinition]:
        """Definition of the grid contained by the tensor, if available

        The grid might not have the same shape as the mappable tensor, but is
        brodcastable to it. This is relatively low level, and should be used
        with caution.
        """
        return self._grid

    def transform(self, affine_transformation: IAffineTransformation) -> "MappableTensor":
        """Apply an affine transformation to the last channel dimension of the tensor

        The affine transformation is not applied right a way, only the
        composition with the existing affine transformation is stored. The
        transformation is applied when the values are generated.
        """
        n_affine_transformation_input_channels = affine_transformation.channels_shape[1] - 1
        if self.channels_shape[-1] != n_affine_transformation_input_channels:
            raise RuntimeError("Affine transformation must have matching input channels")
        if self._grid is None:
            grid: Optional[GridDefinition] = None
        else:
            if not self._grid.is_transformable(affine_transformation):
                return MappableTensor(
                    displacements=self.generate_values(),
                    mask=self._mask,
                    n_channel_dims=self._n_channel_dims,
                    affine_transformation=affine_transformation,
                    grid=None,
                )
            grid = self._grid.broadcast_to_n_channels(
                n_affine_transformation_input_channels
            ).transform(affine_transformation)
        if self._affine_transformation is None:
            assert self._displacements is None
            affine_transformation_on_displacements: Optional[IAffineTransformation] = None
        else:
            affine_transformation_on_displacements = (
                affine_transformation
                @ self._affine_transformation.broadcast_to_n_output_channels(
                    n_affine_transformation_input_channels,
                )
            )
        return MappableTensor(
            displacements=self._displacements,
            mask=self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=affine_transformation_on_displacements,
            grid=grid,
        )

    def __rtruediv__(self, other: Union["MappableTensor", Number, Tensor]) -> "MappableTensor":
        if isinstance(other, (MappableTensor, Tensor, int, float)):
            return self._reciprocal().__mul__(other)
        return NotImplemented

    def _reciprocal(self) -> "MappableTensor":
        return mappable(
            values=1 / self.generate_values(),
            mask=self._mask,
            n_channel_dims=self._n_channel_dims,
        )

    def __truediv__(self, other: Union["MappableTensor", Number, Tensor]) -> "MappableTensor":
        if isinstance(other, MappableTensor):
            return self._truediv_mappable_tensor(other)
        if isinstance(other, (int, float)):
            return self.__mul__(1 / other)
        if isinstance(other, Tensor):
            return self.__mul__(1 / other)
        return NotImplemented

    def _truediv_mappable_tensor(self, other: "MappableTensor") -> "MappableTensor":
        self_values, self_mask = self.generate(generate_missing_mask=False, cast_mask=False)
        other_values, other_mask = other.generate(generate_missing_mask=False, cast_mask=False)
        mask = combine_optional_masks(
            self_mask,
            other_mask,
            n_channel_dims=(self.n_channel_dims, other.n_channel_dims),
        )
        n_channel_dims = max(self.n_channel_dims, other.n_channel_dims)
        self_values, other_values = broadcast_tensors_in_parts(
            self_values, other_values, n_channel_dims=(self.n_channel_dims, other.n_channel_dims)
        )
        return mappable(
            values=self_values / other_values,
            mask=mask,
            n_channel_dims=n_channel_dims,
        )

    def __mul__(self, other: Union["MappableTensor", Number, Tensor]) -> "MappableTensor":
        if isinstance(other, MappableTensor):
            return self._mul_mappable_tensor(other)
        if isinstance(other, (int, float)):
            return self._mul_scalar(other)
        if isinstance(other, Tensor):
            return self._mul_tensor(other)
        return NotImplemented

    def _mul_tensor(self, other: Tensor) -> "MappableTensor":
        if other.ndim > 1:
            raise ValueError(
                "Multiplication is ambigous since n_channel_dims is not specified. "
                "Multiply a MappableTensor instead or apply an affine transformation."
            )
        n_dims = self.channels_shape[-1]
        affine = HostDiagonalAffineTransformation(
            diagonal=other,
            matrix_shape=(n_dims + 1, n_dims + 1),
            dtype=self.dtype,
            device=self.device,
        )
        return self.transform(affine)

    def _mul_scalar(self, other: Number) -> "MappableTensor":
        n_dims = self.channels_shape[-1]
        affine = HostDiagonalAffineTransformation(
            diagonal=other,
            matrix_shape=(n_dims + 1, n_dims + 1),
            dtype=self.dtype,
            device=self.device,
        )
        return self.transform(affine)

    def _mul_mappable_tensor(self, other: "MappableTensor") -> "MappableTensor":
        self_values, self_mask = self.generate(generate_missing_mask=False, cast_mask=False)
        other_values, other_mask = other.generate(generate_missing_mask=False, cast_mask=False)
        mask = combine_optional_masks(
            self_mask,
            other_mask,
            n_channel_dims=(self.n_channel_dims, other.n_channel_dims),
        )
        n_channel_dims = max(self.n_channel_dims, other.n_channel_dims)
        self_values, other_values = broadcast_tensors_in_parts(
            self_values, other_values, n_channel_dims=(self.n_channel_dims, other.n_channel_dims)
        )
        return mappable(
            values=self_values * other_values,
            mask=mask,
            n_channel_dims=n_channel_dims,
        )

    def __add__(self, other: Union["MappableTensor", Number, Tensor]) -> "MappableTensor":
        if isinstance(other, MappableTensor):
            return self._add_mappable_tensor(other)
        if isinstance(other, (int, float)):
            return self._add_scalar(other)
        if isinstance(other, Tensor):
            return self._add_tensor(other)
        return NotImplemented

    def _add_tensor(self, other: Tensor) -> "MappableTensor":
        if other.ndim > 1:
            raise ValueError(
                "Addition is ambigous since n_channel_dims is not specified. "
                "Add a MappableTensor instead or apply an affine transformation."
            )
        n_dims = self.channels_shape[-1]
        affine = HostDiagonalAffineTransformation(
            translation=other,
            matrix_shape=(n_dims + 1, n_dims + 1),
            dtype=self.dtype,
            device=self.device,
        )
        return self.transform(affine)

    def _add_scalar(self, other: Number) -> "MappableTensor":
        n_dims = self.channels_shape[-1]
        affine = HostDiagonalAffineTransformation(
            translation=other,
            matrix_shape=(n_dims + 1, n_dims + 1),
            dtype=self.dtype,
            device=self.device,
        )
        return self.transform(affine)

    def _add_mappable_tensor(self, other: "MappableTensor") -> "MappableTensor":
        mask = combine_optional_masks(
            self.generate_mask(generate_missing_mask=False, cast_mask=False),
            other.generate_mask(generate_missing_mask=False, cast_mask=False),
            n_channel_dims=(self.n_channel_dims, other.n_channel_dims),
        )
        n_channel_dims = max(self.n_channel_dims, other.n_channel_dims)
        if self.grid is not None and other.grid is not None:
            grid: Optional[GridDefinition] = self.grid + other.grid
        elif self.grid is not None:
            grid = self.grid
        else:
            grid = other.grid

        if self.displacements is None:
            displacements = other.displacements
            affine_transformation = other.affine_transformation
        elif other.displacements is None:
            displacements = self.displacements
            affine_transformation = self.affine_transformation
        else:
            assert (
                self.affine_transformation is not None and other.affine_transformation is not None
            )
            self_displacements, other_displacements = broadcast_tensors_in_parts(
                self.affine_transformation(self.displacements),
                other.affine_transformation(other.displacements),
                n_channel_dims=(self.n_channel_dims, other.n_channel_dims),
            )
            displacements = self_displacements + other_displacements
            affine_transformation = None

        return MappableTensor(
            displacements=displacements,
            mask=mask,
            n_channel_dims=n_channel_dims,
            affine_transformation=affine_transformation,
            grid=grid,
        )

    def __sub__(self, other: Union["MappableTensor", Number, Tensor]) -> "MappableTensor":
        return self.__add__(-other)

    def __neg__(self) -> "MappableTensor":
        """Negation of the mappable tensor"""
        return MappableTensor(
            displacements=self._displacements,
            mask=None if self._mask is None else self._mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=(
                None if self._affine_transformation is None else -self._affine_transformation
            ),
            grid=(None if self._grid is None else -self._grid),
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
        """Reduce the tensor to slice on target shape, if possible"""
        if self._grid is None or self._displacements is not None:
            return None
        transformation_matrix = self._grid.affine_transformation.as_host_matrix()
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

    def reduce(self) -> "MappableTensor":
        """Reduce the masked tensor to a plain tensor"""
        values, mask = self.generate(generate_missing_mask=False, cast_mask=False)
        return mappable(
            values=values,
            mask=mask,
            n_channel_dims=self._n_channel_dims,
        )

    def modify_values(
        self, values: Tensor, n_channel_dims: Optional[int] = None
    ) -> "MappableTensor":
        """Modify values of the tensor"""
        if n_channel_dims is None:
            n_channel_dims = self.n_channel_dims
        mask = self._mask
        if mask is not None:
            batch_shape, _, spatial_shape = split_shape(mask.shape)
            mask = mask.view(batch_shape + (1,) * n_channel_dims + spatial_shape)
        return MappableTensor(
            displacements=values,
            mask=mask,
            n_channel_dims=n_channel_dims,
            affine_transformation=None,
            grid=None,
        )

    def mask_and(self, mask: Optional[Tensor]) -> "MappableTensor":
        """Combine mask with logical and"""
        return self.modify_mask(
            combine_optional_masks(self._mask, mask, n_channel_dims=self._n_channel_dims)
        )

    def modify_mask(self, mask: Optional[Tensor]) -> "MappableTensor":
        """Modify mask of the tensor"""
        return MappableTensor(
            displacements=self._displacements,
            mask=mask,
            n_channel_dims=self._n_channel_dims,
            affine_transformation=self._affine_transformation,
            grid=self._grid,
        )

    def __repr__(self) -> str:
        return (
            f"MappableTensor("
            f"displacements={self._displacements}, "
            f"mask={self._mask}, "
            f"n_channel_dims={self._n_channel_dims}, "
            f"affine_transformation={self._affine_transformation}, "
            f"grid={self._grid})"
        )


def mappable(
    values: Tensor, mask: Optional[Tensor] = None, n_channel_dims: int = 1
) -> MappableTensor:
    """Create a mappable tensor from values and mask"""
    return MappableTensor.from_tensor(values, mask=mask, n_channel_dims=n_channel_dims)


def voxel_grid(
    spatial_shape: Sequence[int],
    mask: Optional[Tensor] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> MappableTensor:
    """Create a voxel grid with optional mask"""
    return MappableTensor.create_voxel_grid(
        spatial_shape=spatial_shape, mask=mask, dtype=dtype, device=device
    )
