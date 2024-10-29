"""Mask modifiers"""

from typing import Mapping, Sequence

from torch import Tensor

from .composable_mapping import ComposableMapping
from .dense_deformation import compute_fov_mask_based_on_bounds
from .mappable_tensor import MappableTensor
from .tensor_like import BaseTensorLikeWrapper, ITensorLike


class RectangleMask(BaseTensorLikeWrapper, ComposableMapping):
    """Add values to mask based on bounds"""

    def __init__(
        self,
        min_values: Sequence[float],
        max_values: Sequence[float],
        inclusive_min: bool = True,
        inclusive_max: bool = True,
    ) -> None:
        self._min_values = min_values
        self._max_values = max_values
        self._inclusive_min = inclusive_min
        self._inclusive_max = inclusive_max

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "RectangleMask":
        return RectangleMask(min_values=self._min_values, max_values=self._max_values)

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        values = coordinates.generate_values()
        update_mask = compute_fov_mask_based_on_bounds(
            coordinates=values,
            n_channel_dims=coordinates.n_channel_dims,
            min_values=self._min_values,
            max_values=self._max_values,
            inclusive_min=self._inclusive_min,
            inclusive_max=self._inclusive_max,
        )
        return coordinates.mask_and(update_mask)

    def invert(self, **arguments):
        raise NotImplementedError("Rectangle mask is not invertible")

    def detach(self) -> "RectangleMask":
        return self

    def __repr__(self) -> str:
        return f"RectangleMask(min_values={self._min_values}, max_values={self._max_values})"


class ClearMask(BaseTensorLikeWrapper, ComposableMapping):
    """Clear mask"""

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "ClearMask":
        return ClearMask()

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return masked_coordinates.clear_mask()

    def invert(self, **arguments):
        raise NotImplementedError("Mask clearing is not invertible")

    def detach(self) -> "ClearMask":
        return self

    def __repr__(self) -> str:
        return "ClearMask()"
