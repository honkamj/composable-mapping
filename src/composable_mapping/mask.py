"""Mask modifiers"""

from typing import Mapping, Sequence

from torch import Tensor

from .base import BaseComposableMapping
from .dense_deformation import compute_fov_mask_based_on_bounds
from .mappable_tensor import MappableTensor, PlainTensor
from .tensor_like import ITensorLike
from .util import combine_optional_masks


class RectangleMask(BaseComposableMapping):
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

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        if masked_coordinates.n_channel_dims != 1:
            raise ValueError("Rectangle mask assumes only 1 channel dimension")
        coordinates, mask = masked_coordinates.generate(
            generate_missing_mask=False, cast_mask=False
        )
        update_mask = compute_fov_mask_based_on_bounds(
            coordinates=coordinates,
            min_values=self._min_values,
            max_values=self._max_values,
            inclusive_min=self._inclusive_min,
            inclusive_max=self._inclusive_max,
        )
        updated_mask = combine_optional_masks([mask, update_mask], n_channel_dims=(1, 1))
        return PlainTensor(
            values=coordinates,
            mask=updated_mask,
            n_channel_dims=len(masked_coordinates.channels_shape),
        )

    def invert(self, **inversion_parameters):
        raise NotImplementedError("Rectangle mask is not invertible")

    def detach(self) -> "RectangleMask":
        return self

    def __repr__(self) -> str:
        return f"RectangleMask(min_values={self._min_values}, max_values={self._max_values})"


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

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        return masked_coordinates.clear_mask()

    def invert(self, **inversion_parameters):
        raise NotImplementedError("Mask clearing is not invertible")

    def detach(self) -> "ClearMask":
        return self

    def __repr__(self) -> str:
        return "ClearMask()"
