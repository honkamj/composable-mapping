"""Base classes for composable mapping"""

from abc import abstractmethod
from typing import Mapping, Optional, TypeVar

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

from .interface import IComposableMapping, IMaskedTensor, ITensorLike

BaseTensorLikeWrapperT = TypeVar("BaseTensorLikeWrapperT", bound="BaseTensorLikeWrapper")


class BaseTensorLikeWrapper(ITensorLike):
    """Base tensor wrapper implementation"""

    @abstractmethod
    def _get_tensors(self) -> Mapping[str, Tensor]:
        """Obtain list of wrapped tensors"""

    @abstractmethod
    def _get_children(self) -> Mapping[str, ITensorLike]:
        """Obtain list of tensor like objects contained within the current object"""

    @abstractmethod
    def _modified_copy(
        self: BaseTensorLikeWrapperT,
        tensors: Mapping[str, Tensor],
        children: Mapping[str, ITensorLike],
    ) -> BaseTensorLikeWrapperT:
        """Create a modified copy of the object with new tensors and children"""

    @property
    def dtype(
        self,
    ) -> torch_dtype:
        """Return the dtype of the underlying tensor(s)

        Does not check that all the wrapped tensors have the same dtype.
        """
        tensors = self._get_tensors()
        if not tensors:
            return next(iter(self._get_children().values())).dtype
        return next(iter(tensors.values())).dtype

    @property
    def device(
        self,
    ) -> torch_device:
        """Return the device of the underlying tensor(s)

        Does not check that all the wrapped tensors are on the same device.
        """
        tensors = self._get_tensors()
        if not tensors:
            return next(iter(self._get_children().values())).device
        return next(iter(tensors.values())).device

    def cast(
        self: BaseTensorLikeWrapperT,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> BaseTensorLikeWrapperT:
        modified_tensors = {
            key: tensor.to(dtype=dtype, device=device)
            for key, tensor in self._get_tensors().items()
        }
        modified_children = {
            key: child.cast(dtype=dtype, device=device)
            for key, child in self._get_children().items()
        }
        return self._modified_copy(modified_tensors, modified_children)

    def detach(self: BaseTensorLikeWrapperT) -> BaseTensorLikeWrapperT:
        modified_tensors = {key: tensor.detach() for key, tensor in self._get_tensors().items()}
        modified_children = {key: child.detach() for key, child in self._get_children().items()}
        return self._modified_copy(modified_tensors, modified_children)


class BaseComposableMapping(IComposableMapping, BaseTensorLikeWrapper):
    """Base class for composable mappings"""

    def compose(self, mapping: "IComposableMapping") -> "IComposableMapping":
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

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
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

    def is_identity(self, check_only_if_can_be_done_on_cpu: bool = True) -> bool:
        return self._left_mapping.is_identity(
            check_only_if_can_be_done_on_cpu
        ) and self._right_mapping.is_identity(check_only_if_can_be_done_on_cpu)
