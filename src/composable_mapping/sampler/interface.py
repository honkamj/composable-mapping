"""Interface for samplers"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Optional

from torch import Tensor

from composable_mapping.mappable_tensor.mappable_tensor import MappableTensor

if TYPE_CHECKING:
    from composable_mapping.coordinate_system import CoordinateSystem


class LimitDirection(Enum):
    """Direction of limit"""

    LEFT = "left"
    RIGHT = "right"
    AVERAGE = "average"


class DataFormat:
    """Defines data format for sampled volumes"""

    def __init__(self, representation: str = "coordinates", coordinate_type: str = "world") -> None:
        if representation not in ["coordinates", "displacements"]:
            raise ValueError("Invalid format")
        if coordinate_type not in ["world", "voxel"]:
            raise ValueError("Invalid coordinate type")
        self.representation = representation
        self.coordinate_type = coordinate_type

    @classmethod
    def voxel_displacements(cls) -> "DataFormat":
        """Voxel displacements data format"""
        return cls(representation="displacements", coordinate_type="voxel")

    @classmethod
    def world_displacements(cls) -> "DataFormat":
        """World displacements data format"""
        return cls(representation="displacements", coordinate_type="world")

    @classmethod
    def voxel_coordinates(cls) -> "DataFormat":
        """Voxel coordinates data format"""
        return cls(representation="coordinates", coordinate_type="voxel")

    @classmethod
    def world_coordinates(cls) -> "DataFormat":
        """World coordinates data format"""
        return cls(representation="coordinates", coordinate_type="world")

    def __repr__(self) -> str:
        return (
            f"DataFormat(representation={self.representation}, "
            f"coordinate_type={self.coordinate_type})"
        )


class ISampler(ABC):
    """Samples values on regular grid in voxel coordinates"""

    @abstractmethod
    def __call__(self, volume: MappableTensor, coordinates: MappableTensor) -> MappableTensor:
        """Sample the volume at coordinates

        Args:
            volume: Volume to be interpolated
            coordinates: Interpolation coordinates in voxel coordinates

        Returns:
            Interpolated volume
        """

    @abstractmethod
    def derivative(
        self, spatial_dim: int, limit_direction: LimitDirection = LimitDirection.AVERAGE
    ) -> "ISampler":
        """Return sampler for sampling derivatives corresponding to the current sampler"""

    @abstractmethod
    def inverse(
        self,
        coordinate_system: "CoordinateSystem",
        data_format: DataFormat,
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> "ISampler":
        """Return sampler for sampling inverse values corresponding to the
        current sampler, if available"""

    @abstractmethod
    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        """Interpolate values as tensor"""

    @abstractmethod
    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        """Interpolate mask as tensor"""
