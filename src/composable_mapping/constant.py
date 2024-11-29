"""Returns constant value for all spatial locations"""

from typing import Mapping

from torch import Tensor

from .composable_mapping import ComposableMapping
from .mappable_tensor import MappableTensor, mappable
from .tensor_like import TensorLike
from .util import broadcast_to_in_parts


class Constant(ComposableMapping):
    """Mapping which returns constant value for all spatial locations.

    Arguments.
        value: Tensor with shape (*channels_shape)
    """

    def __init__(
        self,
        value: Tensor,
    ) -> None:
        self._value = value

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {"value": self._value}

    def _get_children(self) -> Mapping[str, TensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, TensorLike]
    ) -> "Constant":
        return Constant(value=tensors["value"])

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        broadcasted_values = broadcast_to_in_parts(
            self._value,
            batch_shape=coordinates.batch_shape,
            spatial_shape=coordinates.spatial_shape,
            n_channel_dims=self._value.ndim,
        )
        return mappable(
            broadcasted_values,
            mask=coordinates.generate_mask(generate_missing_mask=False, cast_mask=False),
            n_channel_dims=self._value.ndim,
        )

    def invert(self, **arguments):
        raise NotImplementedError("Constant mapping is not invertible")

    def __repr__(self) -> str:
        return f"Constant(value={self._value})"
