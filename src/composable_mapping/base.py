"""Base classes for composable mapping"""

from typing import Mapping

from torch import Tensor

from .interface import IComposableMapping
from .mappable_tensor import MappableTensor
from .tensor_like import BaseTensorLikeWrapper, ITensorLike


class BaseComposableMapping(IComposableMapping, BaseTensorLikeWrapper):
    """Base class for composable mappings"""

    def __matmul__(self, mapping: "IComposableMapping") -> "IComposableMapping":
        return _Composition(self, mapping)


class _Composition(BaseComposableMapping):
    """Composition of two mappings"""

    def __init__(self, left_mapping: IComposableMapping, right_mapping: IComposableMapping) -> None:
        self._left_mapping = left_mapping
        self._right_mapping = right_mapping

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_Composition":
        if not isinstance(children["left_mapping"], IComposableMapping) or not isinstance(
            children["right_mapping"], IComposableMapping
        ):
            raise ValueError("Children of a composition must be composable mappings")
        return _Composition(children["left_mapping"], children["right_mapping"])

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"left_mapping": self._left_mapping, "right_mapping": self._right_mapping}

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return self._left_mapping(self._right_mapping(masked_coordinates))

    def invert(self, **inversion_parameters) -> "IComposableMapping":
        return _Composition(
            self._right_mapping.invert(**inversion_parameters),
            self._left_mapping.invert(**inversion_parameters),
        )

    def __repr__(self) -> str:
        return (
            f"_Composition(left_mapping={self._left_mapping}, right_mapping={self._right_mapping})"
        )
