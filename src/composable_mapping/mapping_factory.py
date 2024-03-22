"""Factory methods for generating useful composable mappings"""

from typing import Optional

from torch import Tensor

from composable_mapping.identity import ComposableIdentity
from composable_mapping.sampleable_composable import SamplableComposable

from .affine import AffineTransformation, ComposableAffine
from .grid_mapping import GridCoordinateMapping, GridMappingArgs, GridVolume
from .interface import IComposableMapping, IVoxelCoordinateSystem


def create_composable_affine(transformation_matrix: Tensor) -> IComposableMapping:
    """Create affine composable mapping"""
    return ComposableAffine(AffineTransformation(transformation_matrix))


def create_composable_identity() -> ComposableIdentity:
    """Create identity composable mapping"""
    return ComposableIdentity()


class SamplableComposableFactory:
    """Factory for creating composable mappings which have specific coordinate system attached"""

    def __init__(
        self, coordinate_system: IVoxelCoordinateSystem, grid_mapping_args: GridMappingArgs
    ) -> None:
        self._grid_composable_factory = GridComposableFactory(
            coordinate_system=coordinate_system, grid_mapping_args=grid_mapping_args
        )

    def create_volume(
        self,
        data: Tensor,
        n_channel_dims: int = 1,
        mask: Optional[Tensor] = None,
        is_deformation: bool = False,
    ) -> SamplableComposable:
        """Create samplable volume mapping"""
        return SamplableComposable(
            mapping=self._grid_composable_factory.create_volume(
                data=data, n_channel_dims=n_channel_dims, mask=mask
            ),
            coordinate_system=self._grid_composable_factory.coordinate_system,
            grid_mapping_args=self._grid_composable_factory.grid_mapping_args,
            is_deformation=is_deformation,
        )

    def create_deformation(
        self, displacement_field: Tensor, mask: Optional[Tensor] = None, is_deformation: bool = True
    ) -> SamplableComposable:
        """Create samplable deformation mapping"""
        return SamplableComposable(
            mapping=self._grid_composable_factory.create_deformation(
                displacement_field=displacement_field, mask=mask
            ),
            coordinate_system=self._grid_composable_factory.coordinate_system,
            grid_mapping_args=self._grid_composable_factory.grid_mapping_args,
            is_deformation=is_deformation,
        )

    def create_affine(
        self, transformation_matrix: Tensor, is_deformation: bool = True
    ) -> SamplableComposable:
        """Create samplable affine mapping"""
        return SamplableComposable(
            create_composable_affine(transformation_matrix),
            coordinate_system=self._grid_composable_factory.coordinate_system,
            grid_mapping_args=self._grid_composable_factory.grid_mapping_args,
            is_deformation=is_deformation,
        )

    def create_identity(self, is_deformation: bool = True) -> SamplableComposable:
        """Create samplable identity mapping"""
        return SamplableComposable(
            ComposableIdentity(),
            coordinate_system=self._grid_composable_factory.coordinate_system,
            grid_mapping_args=self._grid_composable_factory.grid_mapping_args,
            is_deformation=is_deformation,
        )


class GridComposableFactory:
    """Factory for creating grid based composable mappings"""

    def __init__(
        self, coordinate_system: IVoxelCoordinateSystem, grid_mapping_args: GridMappingArgs
    ) -> None:
        self.coordinate_system = coordinate_system
        self.grid_mapping_args = grid_mapping_args

    def create_volume(
        self,
        data: Tensor,
        n_channel_dims: int = 1,
        mask: Optional[Tensor] = None,
    ) -> IComposableMapping:
        """Create volume based on grid samples"""
        grid_volume = GridVolume(
            data=data,
            grid_mapping_args=self.grid_mapping_args,
            n_channel_dims=n_channel_dims,
            mask=mask,
        )
        return grid_volume.compose(self.coordinate_system.to_voxel_coordinates)

    def create_deformation(
        self,
        displacement_field: Tensor,
        mask: Optional[Tensor] = None,
    ) -> IComposableMapping:
        """Create mapping based on dense displacement field"""
        grid_volume = GridCoordinateMapping(
            displacement_field=displacement_field,
            grid_mapping_args=self.grid_mapping_args,
            mask=mask,
        )
        return self.coordinate_system.from_voxel_coordinates.compose(grid_volume).compose(
            self.coordinate_system.to_voxel_coordinates
        )
