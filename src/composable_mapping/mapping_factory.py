"""Factory methods for generating useful composable mappings"""

from typing import Optional

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

from composable_mapping.identity import ComposableIdentity
from composable_mapping.sampleable_composable import SamplableComposable

from .affine import AffineTransformation, ComposableAffine
from .grid_mapping import GridCoordinateMapping, GridMappingArgs, GridVolume
from .interface import (
    IComposableMapping,
    ITensorLike,
    IVoxelCoordinateSystem,
    IVoxelCoordinateSystemFactory,
)


def create_composable_affine(transformation_matrix: Tensor) -> IComposableMapping:
    """Create affine composable mapping"""
    return ComposableAffine(AffineTransformation(transformation_matrix))


def create_composable_identity() -> ComposableIdentity:
    """Create identity composable mapping"""
    return ComposableIdentity()


class _BaseComposableFactory:
    def __init__(
        self,
        grid_mapping_args: GridMappingArgs,
        coordinate_system: Optional[IVoxelCoordinateSystem] = None,
        coordinate_system_factory: Optional[IVoxelCoordinateSystemFactory] = None,
    ) -> None:
        if coordinate_system is not None and coordinate_system_factory is not None:
            raise ValueError(
                "Only one of coordinate_system or coordinate_system_factory can be provided"
            )
        if coordinate_system is None and coordinate_system_factory is None:
            raise ValueError(
                "Either coordinate_system or coordinate_system_factory must be provided"
            )
        self.coordinate_system = coordinate_system
        self.coordinate_system_factory = coordinate_system_factory
        self.grid_mapping_args = grid_mapping_args

    def _obtain_coordinate_system(
        self, dtype: Optional[torch_dtype], device: Optional[torch_device]
    ) -> IVoxelCoordinateSystem:
        if self.coordinate_system is None:
            assert (
                self.coordinate_system_factory is not None
            ), "Either coordinate_system or coordinate_system_factory should have been provided"
            return self.coordinate_system_factory.create(dtype=dtype, device=device)
        return self.coordinate_system


class SamplableComposableFactory(_BaseComposableFactory):
    """Factory for creating composable mappings which have specific coordinate system attached"""

    def create_volume(
        self,
        data: Tensor,
        n_channel_dims: int = 1,
        mask: Optional[Tensor] = None,
        is_deformation: bool = False,
    ) -> SamplableComposable:
        """Create samplable volume mapping"""
        coordinate_system = self._obtain_coordinate_system(
            dtype=data.dtype,
            device=data.device,
        )
        return SamplableComposable(
            mapping=create_volume(
                data=data,
                grid_mapping_args=self.grid_mapping_args,
                coordinate_system=coordinate_system,
                n_channel_dims=n_channel_dims,
                mask=mask,
            ),
            coordinate_system=coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
            is_deformation=is_deformation,
        )

    def create_deformation(
        self, displacement_field: Tensor, mask: Optional[Tensor] = None, is_deformation: bool = True
    ) -> SamplableComposable:
        """Create samplable deformation mapping"""
        coordinate_system = self._obtain_coordinate_system(
            dtype=displacement_field.dtype,
            device=displacement_field.device,
        )
        return SamplableComposable(
            mapping=create_deformation(
                displacement_field=displacement_field,
                grid_mapping_args=self.grid_mapping_args,
                coordinate_system=coordinate_system,
                mask=mask,
            ),
            coordinate_system=coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
            is_deformation=is_deformation,
        )

    def create_affine(
        self, transformation_matrix: Tensor, is_deformation: bool = True
    ) -> SamplableComposable:
        """Create samplable affine mapping"""
        return SamplableComposable(
            create_composable_affine(transformation_matrix),
            coordinate_system=self._obtain_coordinate_system(
                dtype=transformation_matrix.dtype, device=transformation_matrix.device
            ),
            grid_mapping_args=self.grid_mapping_args,
            is_deformation=is_deformation,
        )

    def create_identity(
        self,
        is_deformation: bool = True,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> SamplableComposable:
        """Create samplable identity mapping

        Data type and device have effect only if coordinate system factory, not
        coordinate system, is available.
        """
        return SamplableComposable(
            create_composable_identity(),
            coordinate_system=self._obtain_coordinate_system(dtype=dtype, device=device),
            grid_mapping_args=self.grid_mapping_args,
            is_deformation=is_deformation,
        )

    def create_identity_from(
        self,
        reference: ITensorLike,
        is_deformation: bool = True,
    ) -> SamplableComposable:
        """Create samplable identity mapping

        Data type and device for creating the coordinate system are inferred
        from the reference if coordinate system factory is available.
        """
        return SamplableComposable(
            create_composable_identity(),
            coordinate_system=self._obtain_coordinate_system(
                dtype=reference.dtype, device=reference.device
            ),
            grid_mapping_args=self.grid_mapping_args,
            is_deformation=is_deformation,
        )


def create_samplable_identity_from(
    reference: SamplableComposable,
    is_deformation: bool = True,
) -> SamplableComposable:
    """Create samplable identity mapping

    Data type and device for creating the coordinate system are inferred
    from the reference if coordinate system factory is available.
    """
    return SamplableComposable(
        create_composable_identity(),
        coordinate_system=reference.coordinate_system,
        grid_mapping_args=reference.grid_mapping_args,
        is_deformation=is_deformation,
    )


class GridComposableFactory(_BaseComposableFactory):
    """Factory for creating grid based composable mappings"""

    def create_volume(
        self,
        data: Tensor,
        n_channel_dims: int = 1,
        mask: Optional[Tensor] = None,
    ) -> IComposableMapping:
        """Create volume based on grid samples"""
        return create_volume(
            data=data,
            grid_mapping_args=self.grid_mapping_args,
            coordinate_system=self._obtain_coordinate_system(
                dtype=data.dtype,
                device=data.device,
            ),
            n_channel_dims=n_channel_dims,
            mask=mask,
        )

    def create_deformation(
        self,
        displacement_field: Tensor,
        mask: Optional[Tensor] = None,
    ) -> IComposableMapping:
        """Create mapping based on dense displacement field"""
        return create_deformation(
            displacement_field=displacement_field,
            grid_mapping_args=self.grid_mapping_args,
            coordinate_system=self._obtain_coordinate_system(
                dtype=displacement_field.dtype,
                device=displacement_field.device,
            ),
            mask=mask,
        )


def create_volume(
    data: Tensor,
    grid_mapping_args: GridMappingArgs,
    coordinate_system: IVoxelCoordinateSystem,
    n_channel_dims: int = 1,
    mask: Optional[Tensor] = None,
) -> IComposableMapping:
    """Create volume based on grid samples"""
    grid_volume = GridVolume(
        data=data,
        grid_mapping_args=grid_mapping_args,
        n_channel_dims=n_channel_dims,
        mask=mask,
    )
    return grid_volume.compose(coordinate_system.to_voxel_coordinates)


def create_deformation(
    displacement_field: Tensor,
    grid_mapping_args: GridMappingArgs,
    coordinate_system: IVoxelCoordinateSystem,
    mask: Optional[Tensor] = None,
) -> IComposableMapping:
    """Create mapping based on dense displacement field"""
    grid_volume = GridCoordinateMapping(
        displacement_field=displacement_field,
        grid_mapping_args=grid_mapping_args,
        mask=mask,
    )
    return coordinate_system.from_voxel_coordinates.compose(grid_volume).compose(
        coordinate_system.to_voxel_coordinates
    )
