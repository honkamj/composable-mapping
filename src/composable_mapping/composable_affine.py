"""Composable affine transformation"""

from typing import Mapping, Optional

from torch import Tensor

from .affine_transformation import IAffineTransformation
from .base import BaseComposableMapping
from .interface import IComposableMapping
from .mappable_tensor import MappableTensor
from .tensor_like import ITensorLike


class ComposableAffine(BaseComposableMapping):
    """Composable wrapper for affine transformations"""

    def __init__(self, affine_transformation: IAffineTransformation) -> None:
        self.affine_transformation = affine_transformation

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return masked_coordinates.transform(self.affine_transformation)

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"affine_transformation": self.affine_transformation}

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "ComposableAffine":
        if not isinstance(children["affine_transformation"], IAffineTransformation):
            raise ValueError("Child of a composable affine must be an affine transformation")
        return ComposableAffine(children["affine_transformation"])

    def invert(self, **inversion_parameters) -> IComposableMapping:
        return ComposableAffine(self.affine_transformation.invert())

    def __repr__(self) -> str:
        return f"ComposableAffine(affine_transformation={self.affine_transformation})"


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
    """Can be used to trace affine component of a composable mapping"""

    def __init__(self, affine_transformation: Optional[IAffineTransformation] = None) -> None:
        super().__init__(
            spatial_shape=tuple(),
            displacements=None,
            mask=None,
            n_channel_dims=1,
            affine_transformation_on_displacements=None,
            affine_transformation_on_voxel_grid=None,
        )
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
