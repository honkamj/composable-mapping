"""Mask modifiers"""

from typing import Mapping, Sequence

from torch import Tensor

from .base import BaseComposableMapping
from .dense_deformation import compute_fov_mask_based_on_bounds
from .interface import IMaskedTensor, ITensorLike
from .masked_tensor import MaskedTensor
from .util import combine_optional_masks


class RectangleMask(BaseComposableMapping):
    """Add values to mask based on bounds"""

    def __init__(
        self,
        min_values: Sequence[float],
        max_values: Sequence[float],
    ) -> None:
        self._min_values = min_values
        self._max_values = max_values

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "RectangleMask":
        return RectangleMask(min_values=self._min_values, max_values=self._max_values)

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        coordinates, mask = masked_coordinates.generate(generate_missing_mask=False)
        update_mask = compute_fov_mask_based_on_bounds(
            coordinates=coordinates,
            min_values=self._min_values,
            max_values=self._max_values,
            dtype=coordinates.dtype,
        )
        updated_mask = combine_optional_masks([mask, update_mask])
        return MaskedTensor(values=coordinates, mask=updated_mask)

    def invert(self, **inversion_parameters):
        raise NotImplementedError("Rectangle mask is not invertible")

    def detach(self) -> "RectangleMask":
        return self


class ClearMask(BaseComposableMapping):
    """Clear mask"""

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "ClearMask":
        return ClearMask()

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return masked_coordinates.clear_mask()

    def invert(self, **inversion_parameters):
        raise NotImplementedError("Mask clearing is not invertible")

    def detach(self) -> "ClearMask":
        return self
