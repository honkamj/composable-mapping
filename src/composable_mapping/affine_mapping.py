"""Composable affine transformation"""

from typing import Mapping, Optional, Sequence

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

from .composable_mapping import ComposableMapping
from .mappable_tensor import (
    AffineTransformation,
    DiagonalAffineTransformation,
    IAffineTransformation,
    IdentityAffineTransformation,
    MappableTensor,
)
from .tensor_like import BaseTensorLikeWrapper, ITensorLike


class Affine(BaseTensorLikeWrapper, ComposableMapping):
    """Affine transformation"""

    def __init__(self, transformation: IAffineTransformation) -> None:
        self.transformation = transformation

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> "Affine":
        """Create affine mapping from an affine transformation matrix"""
        return cls(AffineTransformation(matrix))

    @classmethod
    def from_diagonal_and_translation(
        cls,
        diagonal: Optional[Tensor] = None,
        translation: Optional[Tensor] = None,
        matrix_shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "Affine":
        """Create affine mapping from diagonal and translation"""
        return cls(
            DiagonalAffineTransformation(
                diagonal=diagonal,
                translation=translation,
                matrix_shape=matrix_shape,
                dtype=dtype,
                device=device,
            )
        )

    @classmethod
    def create_identity(
        cls, n_dims: int, dtype: Optional[torch_dtype] = None, device: Optional[torch_device] = None
    ) -> "Affine":
        """Create identity affine transformation"""
        return cls(IdentityAffineTransformation(n_dims, dtype=dtype, device=device))

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return masked_coordinates.transform(self.transformation)

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"transformation": self.transformation}

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "Affine":
        if not isinstance(children["transformation"], IAffineTransformation):
            raise ValueError("Child of a composable affine must be an affine transformation")
        return Affine(children["transformation"])

    def invert(self, **arguments) -> ComposableMapping:
        return Affine(self.transformation.invert())

    def __repr__(self) -> str:
        return f"Affine(transformation={self.transformation})"


def affine(matrix: Tensor) -> Affine:
    """Create affine transformation from an affine transformation matrix"""
    return Affine.from_matrix(matrix)


def diagonal_affine(
    diagonal: Optional[Tensor] = None,
    translation: Optional[Tensor] = None,
    matrix_shape: Optional[Sequence[int]] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> Affine:
    """Create affine transformation from diagonal and translation"""
    return Affine.from_diagonal_and_translation(
        diagonal=diagonal,
        translation=translation,
        matrix_shape=matrix_shape,
        dtype=dtype,
        device=device,
    )
