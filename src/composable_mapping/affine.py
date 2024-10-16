"""Composable affine transformation"""

from typing import Mapping, Optional, Sequence

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

from composable_mapping.mappable_tensor.affine_transformation import (
    IdentityAffineTransformation,
)

from .base import BaseComposableMapping
from .interface import IComposableMapping
from .mappable_tensor import (
    AffineTransformation,
    DiagonalAffineTransformation,
    IAffineTransformation,
    MappableTensor,
)
from .tensor_like import ITensorLike


class Affine(BaseComposableMapping):
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
    def identity(
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

    def invert(self, **inversion_parameters) -> IComposableMapping:
        return Affine(self.transformation.invert())

    def __repr__(self) -> str:
        return f"Affine(transformation={self.transformation})"


class NotAffineTransformationError(Exception):
    """Error raised when a composable mapping is not affine"""


def as_affine_transformation(
    composable_mapping: IComposableMapping,
) -> IAffineTransformation:
    """Extract affine mapping from composable mapping

    Raises an error if the composable mapping is not fully affine.
    """
    tracer = _AffineTracer()
    traced = composable_mapping(tracer)
    if isinstance(traced, _AffineTracer):
        if traced.traced_affine is None:
            raise NotAffineTransformationError("Could not infer affine transformation")
        return traced.traced_affine
    raise NotAffineTransformationError("Could not infer affine transformation")


class _AffineTracer(MappableTensor):
    # pylint: disable=super-init-not-called
    def __init__(self, affine_transformation: Optional[IAffineTransformation] = None) -> None:
        self.traced_affine: Optional[IAffineTransformation] = affine_transformation

    def transform(self, affine_transformation: IAffineTransformation) -> MappableTensor:
        if self.traced_affine is not None:
            traced_affine = affine_transformation @ self.traced_affine
        else:
            traced_affine = affine_transformation
        return _AffineTracer(traced_affine)

    def __getattribute__(self, name: str):
        if name not in ("transform", "traced_affine"):
            raise NotAffineTransformationError(
                "Could not infer affine transformation since an other operation "
                f"than applying affine transformation was applied ({name})"
            )
        return object.__getattribute__(self, name)
