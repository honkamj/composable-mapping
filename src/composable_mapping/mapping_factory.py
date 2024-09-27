"""Factory methods for generating useful composable mappings"""

from typing import Optional, Union, overload

from torch import Tensor

from .affine_transformation import AffineTransformation
from .composable_affine import ComposableAffine
from .grid_mapping import (
    InterpolationArgs,
    create_deformation,
    create_volume,
    get_interpolation_args,
)
from .identity import ComposableIdentity
from .interface import IComposableMapping
from .mappable_tensor import MappableTensor, PlainTensor
from .samplable_mapping import SamplableDeformationMapping, SamplableVolumeMapping
from .voxel_coordinate_system import (
    IVoxelCoordinateSystemContainer,
    VoxelCoordinateSystem,
)


def create_composable_affine(transformation_matrix: Tensor) -> IComposableMapping:
    """Create affine composable mapping"""
    return ComposableAffine(AffineTransformation(transformation_matrix))


def create_composable_identity() -> ComposableIdentity:
    """Create identity composable mapping"""
    return ComposableIdentity()


class BaseMappingFactory(IVoxelCoordinateSystemContainer):
    """Base class for composable mapping factories

    Implements IVoxelCoordinateSystemFactory interface for it to be usable as an
    argument for resampling operations.
    """

    def __init__(
        self,
        coordinate_system: VoxelCoordinateSystem,
        interpolation_args: Optional[InterpolationArgs] = None,
    ) -> None:
        self._coordinate_system = coordinate_system
        self.interpolation_args = get_interpolation_args(interpolation_args)

    @property
    def coordinate_system(self) -> VoxelCoordinateSystem:
        return self._coordinate_system

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(coordinate_system={self.coordinate_system}, "
            f"interpolation_args={self.interpolation_args})"
        )

    def _handle_tensor_inputs(
        self, data: Union[Tensor, MappableTensor], mask: Optional[Tensor], n_channel_dims: int = 1
    ) -> MappableTensor:
        if isinstance(data, MappableTensor):
            if mask is not None:
                raise ValueError("Mask should not be provided when data is MappableTensor")
            return data
        return PlainTensor(data, mask, n_channel_dims=n_channel_dims)


class SamplableMappingFactory(BaseMappingFactory):
    """Factory for creating composable mappings which have specific coordinate system attached"""

    @overload
    def create_volume(
        self,
        data: MappableTensor,
    ) -> SamplableVolumeMapping: ...

    @overload
    def create_volume(
        self,
        data: Tensor,
        mask: Optional[Tensor] = ...,
        *,
        n_channel_dims: int = ...,
    ) -> SamplableVolumeMapping: ...

    def create_volume(
        self,
        data: Union[Tensor, MappableTensor],
        mask: Optional[Tensor] = None,
        *,
        n_channel_dims: int = 1,
    ) -> SamplableVolumeMapping:
        """Create samplable volume mapping"""
        data = self._handle_tensor_inputs(data, mask, n_channel_dims)
        return SamplableVolumeMapping(
            mapping=create_volume(
                data=data,
                interpolation_args=self.interpolation_args,
                coordinate_system=self.coordinate_system,
            ),
            coordinate_system=self.coordinate_system,
        )

    @overload
    def create_deformation(
        self,
        data: MappableTensor,
        *,
        data_format: str = ...,
        data_coordinates: str = ...,
    ) -> SamplableDeformationMapping: ...

    @overload
    def create_deformation(
        self,
        data: Tensor,
        mask: Optional[Tensor] = ...,
        *,
        data_format: str = ...,
        data_coordinates: str = ...,
    ) -> SamplableDeformationMapping: ...

    def create_deformation(
        self,
        data: Union[Tensor, MappableTensor],
        mask: Optional[Tensor] = None,
        *,
        data_format: str = "displacement",
        data_coordinates: str = "voxel",
    ) -> SamplableDeformationMapping:
        """Create samplable deformation mapping"""
        return SamplableDeformationMapping(
            mapping=create_deformation(
                data=self._handle_tensor_inputs(data, mask),
                interpolation_args=self.interpolation_args,
                coordinate_system=self.coordinate_system,
                data_format=data_format,
                data_coordinates=data_coordinates,
            ),
            coordinate_system=self.coordinate_system,
        )

    def create_affine(
        self,
        transformation_matrix: Tensor,
    ) -> SamplableDeformationMapping:
        """Create samplable affine mapping"""
        return SamplableDeformationMapping(
            create_composable_affine(transformation_matrix),
            coordinate_system=self.coordinate_system,
        )

    def create_identity(
        self,
    ) -> SamplableDeformationMapping:
        """Create samplable identity mapping"""
        return SamplableDeformationMapping(
            create_composable_identity(), coordinate_system=self.coordinate_system
        )


class GridMappingFactory(BaseMappingFactory):
    """Factory for creating grid based composable mappings"""

    @overload
    def create_volume(
        self,
        data: MappableTensor,
    ) -> IComposableMapping: ...

    @overload
    def create_volume(
        self,
        data: Tensor,
        mask: Optional[Tensor] = ...,
        *,
        n_channel_dims: int = ...,
    ) -> IComposableMapping: ...

    def create_volume(
        self,
        data: Union[Tensor, MappableTensor],
        mask: Optional[Tensor] = None,
        *,
        n_channel_dims: int = 1,
    ) -> IComposableMapping:
        """Create volume based on grid samples"""
        data = self._handle_tensor_inputs(data, mask, n_channel_dims)
        return create_volume(
            data=data,
            interpolation_args=self.interpolation_args,
            coordinate_system=self.coordinate_system,
        )

    @overload
    def create_deformation(
        self,
        data: MappableTensor,
        *,
        data_format: str = ...,
        data_coordinates: str = ...,
    ) -> IComposableMapping: ...

    @overload
    def create_deformation(
        self,
        data: Tensor,
        mask: Optional[Tensor] = ...,
        *,
        data_format: str = ...,
        data_coordinates: str = ...,
    ) -> IComposableMapping: ...

    def create_deformation(
        self,
        data: Union[Tensor, MappableTensor],
        mask: Optional[Tensor] = None,
        *,
        data_format: str = "displacement",
        data_coordinates: str = "voxel",
    ) -> IComposableMapping:
        """Create deformation based on regular grid of samples"""
        return create_deformation(
            data=self._handle_tensor_inputs(data, mask),
            interpolation_args=self.interpolation_args,
            coordinate_system=self.coordinate_system,
            data_format=data_format,
            data_coordinates=data_coordinates,
        )
