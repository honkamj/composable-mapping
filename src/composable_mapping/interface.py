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
    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        """Interpolate

        Args:
            volume: Volume to be interpolated with shape
                (batch_size, *channel_dims, dim_1, ..., dim_{n_dims}). Dimension
                order is the same as the coordinate order of the coordinates
            coordinates: Interpolation coordinates with shape
                (batch_size, n_dims, *target_shape)

        Returns:
            Interpolated volume with shape (batch_size, *channel_dims, *target_shape)
        """
