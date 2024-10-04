"""Interfaces for composable mapping"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence

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
        self, volume: Tensor, voxel_coordinates: Tensor, n_channel_dims: int = 1
    ) -> Tensor:
        """Interpolate values"""

    @abstractmethod
    def interpolate_mask(
        self,
        mask: Optional[Tensor],
        voxel_coordinates: Tensor,
        spatial_shape: Optional[Sequence[int]] = None,
        n_channel_dims: int = 1,
    ) -> Optional[Tensor]:
        """Interpolate mask"""
