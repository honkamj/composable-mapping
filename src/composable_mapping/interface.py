"""Interfaces for composable mapping"""

from abc import ABC, abstractmethod

from torch import Tensor

from .mappable_tensor import MappableTensor
from .tensor_like import ITensorLike


class IComposableMapping(ITensorLike):
    """Composable mapping"""

    @abstractmethod
    def __matmul__(self, mapping: "IComposableMapping") -> "IComposableMapping":
        """Compose with another mapping"""

    @abstractmethod
    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        """Evaluate the mapping at coordinates"""

    @abstractmethod
    def invert(self, **inversion_parameters) -> "IComposableMapping":
        """Invert the mapping

        Args:
            inversion_parameters: Possible inversion parameters
        """


class IInterpolator(ABC):
    """Interpolates values on regular grid in voxel coordinates"""

    @abstractmethod
    def __call__(self, volume: MappableTensor, coordinates: MappableTensor) -> MappableTensor:
        """Interpolate

        Args:
            volume: Volume to be interpolated
            coordinates: Interpolation coordinates

        Returns:
            Interpolated volume
        """

    @abstractmethod
    def interpolate_values(
        self,
        values: Tensor,
        voxel_coordinates: Tensor,
    ) -> Tensor:
        """Interpolate values as tensor"""

    @abstractmethod
    def interpolate_mask(
        self,
        mask: Tensor,
        voxel_coordinates: Tensor,
    ) -> Tensor:
        """Interpolate mask as tensor"""
