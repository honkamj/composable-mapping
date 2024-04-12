"""Factory methods for generating useful composable mappings"""

from typing import Optional, Union, overload

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

from composable_mapping.identity import ComposableIdentity
from composable_mapping.masked_tensor import MaskedTensor
from composable_mapping.samplable_mapping import (
    BaseSamplableMapping,
    SamplableDeformationMapping,
    SamplableVolumeMapping,
)

from .affine import AffineTransformation, ComposableAffine
from .grid_mapping import (
    GridMappingArgs,
    create_deformation_from_voxel_data,
    create_deformation_from_world_data,
    create_volume,
)
from .interface import (
    IComposableMapping,
    IMaskedTensor,
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


class BaseComposableFactory:
    """Base class for composable mapping factories"""

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

    def obtain_coordinate_system(
        self, dtype: Optional[torch_dtype], device: Optional[torch_device]
    ) -> IVoxelCoordinateSystem:
        """Obtain coordinate system"""
        if self.coordinate_system is None:
            if self.coordinate_system_factory is None:
                raise ValueError(
                    "Either coordinate_system or coordinate_system_factory must be provided"
                )
            return self.coordinate_system_factory.create(dtype=dtype, device=device)
        return self.coordinate_system

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(coordinate_system={self.coordinate_system}, "
            f"coordinate_system_factory={self.coordinate_system_factory}, "
            f"grid_mapping_args={self.grid_mapping_args})"
        )

    def _handle_tensor_inputs(
        self, data: Union[Tensor, IMaskedTensor], mask: Optional[Tensor]
    ) -> IMaskedTensor:
        if isinstance(data, IMaskedTensor):
            if mask is not None:
                raise ValueError("Mask should not be provided when data is IMaskedTensor")
            return data
        return MaskedTensor(data, mask)


class SamplableComposableFactory(BaseComposableFactory):
    """Factory for creating composable mappings which have specific coordinate system attached"""

    @overload
    def create_volume(
        self,
        data: IMaskedTensor,
        *,
        n_channel_dims: int = ...,
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
        data: Union[Tensor, IMaskedTensor],
        mask: Optional[Tensor] = None,
        *,
        n_channel_dims: int = 1,
    ) -> SamplableVolumeMapping:
        """Create samplable volume mapping"""
        data = self._handle_tensor_inputs(data, mask)
        coordinate_system = self.obtain_coordinate_system(
            dtype=data.dtype,
            device=data.device,
        )
        return SamplableVolumeMapping(
            mapping=create_volume(
                data=data,
                grid_mapping_args=self.grid_mapping_args,
                coordinate_system=coordinate_system,
                n_channel_dims=n_channel_dims,
            ),
            coordinate_system=coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
        )

    @overload
    def create_deformation(
        self,
        data: IMaskedTensor,
        *,
        data_format: str = ...,
        data_coordinates: str = ...,
        resample_as: Optional[str] = ...,
    ) -> SamplableDeformationMapping: ...

    @overload
    def create_deformation(
        self,
        data: Tensor,
        mask: Optional[Tensor] = ...,
        *,
        data_format: str = ...,
        data_coordinates: str = ...,
        resample_as: Optional[str] = ...,
    ) -> SamplableDeformationMapping: ...

    def create_deformation(
        self,
        data: Union[Tensor, IMaskedTensor],
        mask: Optional[Tensor] = None,
        *,
        data_format: str = "displacement_field",
        data_coordinates: str = "voxel",
        resample_as: Optional[str] = None,
    ) -> SamplableDeformationMapping:
        """Create samplable deformation mapping"""
        if resample_as is None:
            resample_as = data_format
        data = self._handle_tensor_inputs(data, mask)
        coordinate_system = self.obtain_coordinate_system(
            dtype=data.dtype,
            device=data.device,
        )
        if data_coordinates == "voxel":
            factory = create_deformation_from_voxel_data
        elif data_coordinates == "world":
            factory = create_deformation_from_world_data
        else:
            raise ValueError(f"Unsupported data coordinates: {data_coordinates}")
        return SamplableDeformationMapping(
            mapping=factory(
                data=data,
                grid_mapping_args=self.grid_mapping_args,
                coordinate_system=coordinate_system,
                data_format=data_format,
            ),
            coordinate_system=coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
            resample_as=data_format,
        )

    def create_affine(
        self, transformation_matrix: Tensor, *, resample_as: str = "displacement_field"
    ) -> SamplableDeformationMapping:
        """Create samplable affine mapping"""
        return SamplableDeformationMapping(
            create_composable_affine(transformation_matrix),
            coordinate_system=self.obtain_coordinate_system(
                dtype=transformation_matrix.dtype, device=transformation_matrix.device
            ),
            grid_mapping_args=self.grid_mapping_args,
            resample_as=resample_as,
        )

    def create_identity(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
        *,
        resample_as: str = "displacement_field",
    ) -> SamplableDeformationMapping:
        """Create samplable identity mapping

        Data type and device have effect only if coordinate system factory, not
        coordinate system, is available.
        """
        return SamplableDeformationMapping(
            create_composable_identity(),
            coordinate_system=self.obtain_coordinate_system(dtype=dtype, device=device),
            grid_mapping_args=self.grid_mapping_args,
            resample_as=resample_as,
        )

    def create_identity_from(
        self,
        reference: Union[ITensorLike, Tensor],
        *,
        resample_as: str = "displacement_field",
    ) -> SamplableDeformationMapping:
        """Create samplable identity mapping

        Data type and device for creating the coordinate system are inferred
        from the reference if coordinate system factory is available.
        """
        return SamplableDeformationMapping(
            create_composable_identity(),
            coordinate_system=self.obtain_coordinate_system(
                dtype=reference.dtype, device=reference.device
            ),
            grid_mapping_args=self.grid_mapping_args,
            resample_as=resample_as,
        )


def create_samplable_identity_from(
    reference: BaseSamplableMapping,
    *,
    resample_as: str = "displacement_field",
) -> SamplableDeformationMapping:
    """Create samplable identity mapping

    Data type and device for creating the coordinate system are inferred
    from the reference if coordinate system factory is available.
    """
    return SamplableDeformationMapping(
        create_composable_identity(),
        coordinate_system=reference.coordinate_system,
        grid_mapping_args=reference.grid_mapping_args,
        resample_as=resample_as,
    )


class GridComposableFactory(BaseComposableFactory):
    """Factory for creating grid based composable mappings"""

    @overload
    def create_volume(
        self,
        data: IMaskedTensor,
        *,
        n_channel_dims: int = ...,
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
        data: Union[Tensor, IMaskedTensor],
        mask: Optional[Tensor] = None,
        *,
        n_channel_dims: int = 1,
    ) -> IComposableMapping:
        """Create volume based on grid samples"""
        data = self._handle_tensor_inputs(data, mask)
        return create_volume(
            data=data,
            grid_mapping_args=self.grid_mapping_args,
            coordinate_system=self.obtain_coordinate_system(
                dtype=data.dtype,
                device=data.device,
            ),
            n_channel_dims=n_channel_dims,
        )

    @overload
    def create_deformation(
        self,
        data: IMaskedTensor,
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
        data: Union[Tensor, IMaskedTensor],
        mask: Optional[Tensor] = None,
        *,
        data_format: str = "displacement_field",
        data_coordinates: str = "voxel",
    ) -> IComposableMapping:
        """Create deformation based on regular grid of samples"""
        if data_coordinates == "voxel":
            factory = create_deformation_from_voxel_data
        elif data_coordinates == "world":
            factory = create_deformation_from_world_data
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        data = self._handle_tensor_inputs(data, mask)
        return factory(
            data=data,
            grid_mapping_args=self.grid_mapping_args,
            coordinate_system=self.obtain_coordinate_system(
                dtype=data.dtype,
                device=data.device,
            ),
            data_format=data_format,
        )
