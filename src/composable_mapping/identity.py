"""Identity mapping"""

from typing import Mapping

from torch import Tensor

from .base import BaseComposableMapping
from .mappable_tensor import MappableTensor
from .tensor_like import ITensorLike


class ComposableIdentity(BaseComposableMapping):
    """Identity mapping"""

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "ComposableIdentity":
        return ComposableIdentity()

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return masked_coordinates

    def invert(self, **_inversion_parameters) -> "ComposableIdentity":
        return ComposableIdentity()

    def detach(self) -> "ComposableIdentity":
        return self

    def __repr__(self) -> str:
        return "ComposableIdentity()"
