""""Composable mapping bundled together with a coordinate system"""

from itertools import combinations
from typing import Any, Callable, Mapping, Optional, Tuple, TypeVar, Union

from matplotlib.figure import Figure  # type: ignore
from matplotlib.pyplot import subplots  # type: ignore
from torch import Tensor

from composable_mapping.base import ComposableMappingDecorator, Identity
from composable_mapping.finite_difference import (
    SpatialDerivationArguments,
    SpatialJacobiansArguments,
)

from .coordinate_system import CoordinateSystem, ICoordinateSystemContainer
from .grid_mapping import VoxelGridDeformation, VoxelGridVolume
from .interface import (
    IComposableMapping,
    IDeformationComposableMapping,
    IGenericComposableMapping,
    IGridDeformationComposableMapping,
    IGridVolumeComposableMapping,
    ISampler,
    Number,
)
from .mappable_tensor import MappableTensor, mappable


class DeformationDecorator(ComposableMappingDecorator, IDeformationComposableMapping):
    """Decorator for deformation composable mappings"""

    def _handle_operator(
        self,
        other: Optional[Union["IComposableMapping", MappableTensor, Number, Tensor]],
        operator: Callable[[IComposableMapping, Any], IComposableMapping],
    ) -> IComposableMapping:
        if isinstance(other, IGridDeformationComposableMapping):
            return GridDeformationDecorator(
                operator(self.mapping, other.mapping), other.coordinate_system
            )
        if isinstance(other, IGridVolumeComposableMapping):
            return GridVolumeDecorator(
                operator(self.mapping, other.mapping), other.coordinate_system
            )
        if isinstance(other, IDeformationComposableMapping):
            return type(self)(operator(self.mapping, other.mapping))
        return GenericDecorator(operator(self.mapping, other.mapping))

    def sample_to_as_displacement_field(
        self,
        target: ICoordinateSystemContainer,
        *,
        data_coordinates: str = "voxel",
    ) -> MappableTensor:
        coordinate_system = target.coordinate_system
        data = self.sample_to(target)
        if data_coordinates == "world":
            return data - coordinate_system.grid()
        if data_coordinates == "voxel":
            return coordinate_system.to_voxel_coordinates(data) - coordinate_system.voxel_grid()
        raise ValueError(f"Invalid option for data coordinates: {data_coordinates}")

    def resample_to(
        self,
        target: ICoordinateSystemContainer,
        sampler: Optional["ISampler"] = None,
    ) -> IGridDeformationComposableMapping:
        return GridDeformationDecorator.from_mappable_tensor(
            self.sample_to_as_displacement_field(target), target.coordinate_system, sampler=sampler
        )


class GenericDecorator(ComposableMappingDecorator, IGenericComposableMapping):
    """Decorator for deformation composable mappings"""

    def _handle_operator(
        self,
        other: Optional[Union["IComposableMapping", MappableTensor, Number, Tensor]],
        operator: Callable[[IComposableMapping, Any], IComposableMapping],
    ) -> IComposableMapping:
        if isinstance(other, (IGridDeformationComposableMapping, IGridVolumeComposableMapping)):
            return GridVolumeDecorator(
                operator(self.mapping, other.mapping), other.coordinate_system
            )
        return GenericDecorator(operator(self.mapping, other.mapping))

    def resample_to(
        self,
        target: ICoordinateSystemContainer,
        sampler: Optional["ISampler"] = None,
    ) -> IGridVolumeComposableMapping:
        return GridVolumeDecorator.from_mappable_tensor(
            self.sample_to(target), target.coordinate_system, sampler=sampler
        )


class BaseGridDecorator(IComposableMapping, ICoordinateSystemContainer):
    """Base implementation for composable mappings bundled together with a coordinate system

    Arguments:
        mapping: Composable mapping to be wrapped
        coordinate_system: Coordinate system to use for sampling and resampling
    """

    def sample(self) -> MappableTensor:
        """Sample the mapping"""
        return self.sample_to(self)

    def estimate_spatial_derivatives(
        self,
        spatial_dim: int,
        sampler: Optional[ISampler] = None,
        arguments: Optional[SpatialDerivationArguments] = None,
    ) -> IGridVolumeComposableMapping:
        """Estimate spatial derivatives over the mapping at the grid locations

        Args:
            spatial_dim: Spatial dimension to estimate the derivative for
            other_dims: How to handle the other dimensions, see
                finite_difference.estimate_spatial_derivatives for more details
            central: Whether to use central differences
            sampler: Sampler to use for the generated volume of derivatives
        """
        return self.estimate_spatial_derivatives_to(
            self,
            spatial_dim=spatial_dim,
            sampler=sampler,
            arguments=arguments,
        )

    def estimate_spatial_jacobian_matrices(
        self,
        sampler: Optional[ISampler] = None,
        arguments: Optional[SpatialJacobiansArguments] = None,
    ) -> IGridVolumeComposableMapping:
        """Estimate spatial Jacobian matrices over the mapping at the grid locations

        Args:
            central: Whether to use central differences
            sampler: Sampler to use for the generated volume of Jacobian matrices
        """
        return self.estimate_spatial_jacobian_matrices_to(
            self, sampler=sampler, arguments=arguments
        )


class GridDeformationDecorator(
    DeformationDecorator, BaseGridDecorator, IGridDeformationComposableMapping
):
    """Composable deformation with an assigned coordinate system"""

    def __init__(
        self,
        mapping: IComposableMapping,
        coordinate_system: CoordinateSystem,
    ):
        super().__init__(mapping)
        self._coordinate_system = coordinate_system

    def _handle_operator(
        self,
        other: Optional[Union["IComposableMapping", MappableTensor, Number, Tensor]],
        operator: Callable[[IComposableMapping, Any], IComposableMapping],
    ) -> IComposableMapping:
        TODO

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return self._coordinate_system

    @classmethod
    def create_identity(cls, coordinate_system: CoordinateSystem) -> "GridDeformationDecorator":
        """Create an identity deformation with the given coordinate system"""
        return cls(mapping=Identity(), coordinate_system=coordinate_system)

    @classmethod
    def from_tensor(
        cls,
        data: Tensor,
        coordinate_system: CoordinateSystem,
        mask: Optional[Tensor] = None,
        n_channel_dims: int = 1,
        sampler: Optional[ISampler] = None,
        data_format: str = "displacement",
        data_coordinates: str = "voxel",
    ) -> IGridDeformationComposableMapping:
        """Create a deformation from a tensor"""
        return cls.from_mappable_tensor(
            mappable(data, mask, n_channel_dims=n_channel_dims),
            coordinate_system=coordinate_system,
            sampler=sampler,
            data_format=data_format,
            data_coordinates=data_coordinates,
        )

    @classmethod
    def from_mappable_tensor(
        cls,
        data: MappableTensor,
        coordinate_system: CoordinateSystem,
        *,
        sampler: Optional[ISampler] = None,
        data_format: str = "displacement",
        data_coordinates: str = "voxel",
    ) -> IGridDeformationComposableMapping:
        """Create a deformation from a tensor"""
        if data_coordinates == "world":
            if data_format == "displacement":
                data = data + coordinate_system.grid()
                data_format = "coordinate"
            data = coordinate_system.to_voxel_coordinates(data)
        elif data_coordinates != "voxel":
            raise ValueError(f"Invalid data_coordinates: {data_coordinates}")
        if data_format == "coordinate":
            data = data - coordinate_system.voxel_grid()
        elif data_format != "displacement":
            raise ValueError(f"Invalid data_format: {data_format}")
        if data.spatial_shape != coordinate_system.shape:
            raise ValueError("Spatial shape of data must match shape of coordinate system")
        return cls(
            mapping=(
                coordinate_system.from_voxel_coordinates
                @ VoxelGridDeformation(data, sampler)
                @ coordinate_system.to_voxel_coordinates
            ),
            coordinate_system=coordinate_system,
        )

    def resample(
        self,
        sampler: Optional["ISampler"] = None,
    ) -> IGridDeformationComposableMapping:
        return self.resample_to(self, sampler=sampler)


class GridVolumeDecorator(GenericDecorator, BaseGridDecorator, IGridVolumeComposableMapping):
    """Composable continuous volume with an assigned coordinate system"""

    def __init__(
        self,
        mapping: IComposableMapping,
        coordinate_system: CoordinateSystem,
    ):
        super().__init__(mapping)
        self._coordinate_system = coordinate_system

    def _handle_operator(
        self,
        other: Optional[Union["IComposableMapping", MappableTensor, Number, Tensor]],
        operator: Callable[[IComposableMapping, Any], IComposableMapping],
    ) -> IComposableMapping:
        TODO

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return self._coordinate_system

    @classmethod
    def from_tensor(
        cls,
        data: Tensor,
        coordinate_system: CoordinateSystem,
        mask: Optional[Tensor] = None,
        n_channel_dims: int = 1,
        sampler: Optional[ISampler] = None,
    ) -> IGridVolumeComposableMapping:
        """Create a volume from a tensor"""
        return cls.from_mappable_tensor(
            mappable(data, mask, n_channel_dims=n_channel_dims),
            coordinate_system,
            sampler=sampler,
        )

    @classmethod
    def from_mappable_tensor(
        cls,
        data: MappableTensor,
        coordinate_system: CoordinateSystem,
        sampler: Optional[ISampler] = None,
    ) -> IGridVolumeComposableMapping:
        """Create a volume from a tensor"""
        if data.spatial_shape != coordinate_system.shape:
            raise ValueError("Spatial shape of data must match shape of coordinate system")
        return cls(
            mapping=VoxelGridVolume(data, sampler) @ coordinate_system.to_voxel_coordinates,
            coordinate_system=coordinate_system,
        )

    def resample(
        self,
        sampler: Optional["ISampler"] = None,
    ) -> IGridVolumeComposableMapping:
        return self.resample_to(self, sampler=sampler)
