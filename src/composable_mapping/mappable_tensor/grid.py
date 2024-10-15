"""Defines a voxel grid with given shape transformed by given affine transformation"""

from typing import Mapping, Sequence, Tuple

from torch import Tensor, broadcast_shapes, zeros

from composable_mapping.dense_deformation import generate_voxel_coordinate_grid
from composable_mapping.tensor_like import BaseTensorLikeWrapper, ITensorLike
from composable_mapping.util import (
    get_batch_shape,
    get_channels_shape,
    has_spatial_dims,
    is_broadcastable,
)

from .affine_transformation import IAffineTransformation


class Grid(BaseTensorLikeWrapper):
    """A voxel grid with given shape transformed by given affine transformation"""

    def __init__(
        self,
        spatial_shape: Sequence[int],
        affine_transformation: IAffineTransformation,
    ) -> None:
        self._spatial_shape = spatial_shape
        if has_spatial_dims(affine_transformation.shape, n_channel_dims=2):
            raise ValueError(
                "Affine transformation for grid may not differ over spatial dimensions"
            )
        self._affine_transformation = affine_transformation

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"affine_transformation": self._affine_transformation}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "Grid":
        if "affine_transformation" in children or not isinstance(
            children["affine_transformation"], IAffineTransformation
        ):
            raise ValueError("Invalid children for grid")
        return Grid(
            spatial_shape=self._spatial_shape,
            affine_transformation=children["affine_transformation"],
        )

    def generate(self) -> Tensor:
        """Generate the grid"""
        if self._affine_transformation.is_zero():
            return zeros(self.shape, dtype=self.dtype, device=self.device)
        voxel_grid = generate_voxel_coordinate_grid(
            shape=self._spatial_shape, device=self.device, dtype=self.dtype
        )
        return self._affine_transformation(voxel_grid, n_channel_dims=1)

    @property
    def affine_transformation(self) -> IAffineTransformation:
        """Get the affine transformation"""
        return self._affine_transformation

    @property
    def spatial_shape(self) -> Tuple[int, ...]:
        """Shape of the spatial dimensions"""
        return tuple(self._spatial_shape)

    @property
    def channels_shape(self) -> Tuple[int, ...]:
        """Shape of the channel dimensions"""
        return get_channels_shape(self.shape)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Shape of the batch dimensions"""
        return get_batch_shape(self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the grid"""
        return self._affine_transformation.get_output_shape(
            (1, len(self._spatial_shape)) + tuple(self._spatial_shape)
        )

    def is_transformable(self, affine_transformation: IAffineTransformation) -> bool:
        """Check if the grid is transformable by the given affine transformation"""
        if has_spatial_dims(affine_transformation.shape):
            return False
        if not is_broadcastable(
            affine_transformation.batch_shape, self._affine_transformation.batch_shape
        ):
            return False
        return (
            affine_transformation.channels_shape[1] == self._affine_transformation.channels_shape[0]
        )

    def __add__(self, other: "Grid") -> "Grid":
        """Sum two grids"""
        if self.spatial_shape != other.spatial_shape:
            raise RuntimeError("Grids can not be added")
        broadcasted_n_channel_dims = broadcast_shapes(self.channels_shape, other.channels_shape)[-1]
        self_affine = broadcast_affine_to_n_input_channels(
            self.affine_transformation, n_input_channels=broadcasted_n_channel_dims
        )
        other_affine = broadcast_affine_to_n_input_channels(
            other.affine_transformation, n_input_channels=broadcasted_n_channel_dims
        )
        return Grid(
            spatial_shape=self.spatial_shape,
            affine_transformation=self_affine + other_affine,
        )

    def __neg__(self) -> "Grid":
        """Negate the grid"""
        return Grid(
            spatial_shape=self.spatial_shape,
            affine_transformation=-self.affine_transformation,
        )

    def __sub__(self, other: "Grid") -> "Grid":
        """Subtract two grids"""
        return self + (-other)
