"""Tensor like interface and base class"""

from abc import ABC, abstractmethod
from typing import Mapping, Optional, TypeVar

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

ITensorLikeT = TypeVar("ITensorLikeT", bound="ITensorLike")


class ITensorLike(ABC):
    """Interface for classes having tensor like properties

    Usually contains wrapped tensors with device, dtype, and detachment
    related functionalities corresponding directly to the wrapped tensors.
    """

    @property
    @abstractmethod
    def dtype(
        self,
    ) -> torch_dtype:
        """Return the dtype of the underlying tensor(s)"""

    @property
    @abstractmethod
    def device(
        self,
    ) -> torch_device:
        """Return the device of the underlying tensor(s)"""

    @abstractmethod
    def cast(
        self: ITensorLikeT,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
        non_blocking: bool = False,
    ) -> ITensorLikeT:
        """Cast underlying tensors to given dtype and device

        We will not use method name "to" as it would create conflict with
        torch.nn.Module.to method which does casting in-place.
        """

    @abstractmethod
    def detach(self: ITensorLikeT) -> ITensorLikeT:
        """Detach the wrapped tensors from computational graph"""

    @abstractmethod
    def pin_memory(self: ITensorLikeT) -> ITensorLikeT:
        """Move the wrapped tensors to page-locked memory"""


BaseTensorLikeWrapperT = TypeVar("BaseTensorLikeWrapperT", bound="BaseTensorLikeWrapper")


class BaseTensorLikeWrapper(ITensorLike):
    """Base tensor wrapper implementation"""

    def _get_tensors(self) -> Mapping[str, Tensor]:
        """Obtain list of wrapped tensors"""
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        """Obtain list of tensor like objects contained within the current object"""
        return {}

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
        non_blocking: bool = False,
    ) -> BaseTensorLikeWrapperT:
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        modified_tensors = {
            key: tensor.to(dtype=dtype, device=device, non_blocking=non_blocking)
            for key, tensor in self._get_tensors().items()
        }
        modified_children = {
            key: child.cast(dtype=dtype, device=device, non_blocking=non_blocking)
            for key, child in self._get_children().items()
        }
        return self._modified_copy(modified_tensors, modified_children)

    def pin_memory(self: BaseTensorLikeWrapperT) -> BaseTensorLikeWrapperT:
        modified_tensors = {key: tensor.pin_memory() for key, tensor in self._get_tensors().items()}
        modified_children = {key: child.pin_memory() for key, child in self._get_children().items()}
        return self._modified_copy(modified_tensors, modified_children)

    def detach(self: BaseTensorLikeWrapperT) -> BaseTensorLikeWrapperT:
        modified_tensors = {key: tensor.detach() for key, tensor in self._get_tensors().items()}
        modified_children = {key: child.detach() for key, child in self._get_children().items()}
        return self._modified_copy(modified_tensors, modified_children)
