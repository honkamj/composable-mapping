"""Identity mapping"""

from typing import Mapping

from torch import Tensor

from .base import BaseComposableMapping
from .mappable_tensor import MappableTensor
from .tensor_like import ITensorLike


class Identity(BaseComposableMapping):
    """Identity mapping"""

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "Identity":
        return Identity()

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return masked_coordinates

    def invert(self, **_inversion_parameters) -> "Identity":
        return Identity()

    def detach(self) -> "Identity":
        return self

    def __repr__(self) -> str:
        return "Identity()"
