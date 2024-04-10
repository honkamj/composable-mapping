""""Composable mapping bundled together with a coordinate system"""

from itertools import combinations
from typing import Any, Mapping, Optional, Tuple, TypeVar, Union

from matplotlib.figure import Figure  # type: ignore
from matplotlib.pyplot import subplots  # type: ignore
from torch import Tensor

from composable_mapping.finite_difference import (
    estimate_spatial_derivatives_for_mapping,
    estimate_spatial_jacobian_matrices_for_mapping,
)

from .affine import ComposableAffine
from .base import BaseTensorLikeWrapper
from .grid_mapping import (
    GridMappingArgs,
    GridVolume,
    create_deformation_from_voxel_data,
)
from .interface import (
    IAffineTransformation,
    IComposableMapping,
    IMaskedTensor,
    ITensorLike,
    IVoxelCoordinateSystem,
)

BaseSamplableMappingT = TypeVar("BaseSamplableMappingT", bound="BaseSamplableMapping")


class BaseSamplableMapping(BaseTensorLikeWrapper):
    """Base implementation for composable mappings bundled together with a coordinate system

    Arguments:
        mapping: Composable mapping to be wrapped
        coordinate_system: Coordinate system to use for sampling and resampling
        grid_mapping_args: Grid mapping arguments to use for resampling
    """

    def __init__(
        self,
        mapping: IComposableMapping,
        coordinate_system: IVoxelCoordinateSystem,
        grid_mapping_args: GridMappingArgs,
    ):
        self.mapping = mapping
        self.coordinate_system = coordinate_system
        self.grid_mapping_args = grid_mapping_args

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return self.mapping(masked_coordinates)

    def sample(self) -> IMaskedTensor:
        """Sample the mapping"""
        return self.mapping(self.coordinate_system.grid)

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"mapping": self.mapping, "coordinate_system": self.coordinate_system}

    def invert(self: BaseSamplableMappingT, **inversion_parameters) -> BaseSamplableMappingT:
        """Invert the mapping"""
        return self._simplified_modified_copy(
            mapping=self.mapping.invert(**inversion_parameters),
            coordinate_system=self.coordinate_system,
        )

    def _simplified_modified_copy(
        self: BaseSamplableMappingT,
        mapping: IComposableMapping,
        coordinate_system: IVoxelCoordinateSystem,
    ) -> BaseSamplableMappingT:
        raise NotImplementedError

    def _modified_copy(
        self: BaseSamplableMappingT,
        tensors: Mapping[str, Tensor],
        children: Mapping[str, ITensorLike],
    ) -> BaseSamplableMappingT:
        if not isinstance(children["mapping"], IComposableMapping) or not isinstance(
            children["coordinate_system"], IVoxelCoordinateSystem
        ):
            raise ValueError("Invalid children for samplable mapping")
        return self._simplified_modified_copy(
            mapping=children["mapping"],
            coordinate_system=children["coordinate_system"],
        )

    def __matmul__(
        self: BaseSamplableMappingT,
        right_mapping: Union[
            "SamplableVolumeMapping",
            "SamplableDeformationMapping",
            IComposableMapping,
            IAffineTransformation,
        ],
    ) -> BaseSamplableMappingT:
        if isinstance(right_mapping, IAffineTransformation):
            right_composable_mapping: IComposableMapping = ComposableAffine(right_mapping)
            coordinate_system = self.coordinate_system
        elif isinstance(right_mapping, IComposableMapping):
            right_composable_mapping = right_mapping
            coordinate_system = self.coordinate_system
        elif isinstance(right_mapping, SamplableVolumeMapping) or isinstance(
            right_mapping, SamplableDeformationMapping
        ):
            right_composable_mapping = right_mapping.mapping
            coordinate_system = right_mapping.coordinate_system
        else:
            return NotImplemented
        return self._simplified_modified_copy(
            mapping=self.mapping.compose(right_composable_mapping),
            coordinate_system=coordinate_system,
        )

    def __rmatmul__(
        self: BaseSamplableMappingT, left_mapping: IAffineTransformation
    ) -> BaseSamplableMappingT:
        if not isinstance(left_mapping, IAffineTransformation):
            return NotImplemented
        return self._simplified_modified_copy(
            mapping=ComposableAffine(left_mapping).compose(self.mapping),
            coordinate_system=self.coordinate_system,
        )

    def estimate_spatial_derivatives(
        self,
        spatial_dim: int,
        other_dims: Optional[str] = None,
        central: bool = False,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        """Estimate spatial derivatives over the mapping at the grid locations

        Args:
            spatial_dim: Spatial dimension to estimate the derivative for
            other_dims: How to handle the other dimensions, see
                finite_difference.estimate_spatial_derivatives for more details
            central: Whether to use central differences
            out: Output tensor to store the result
        """
        return estimate_spatial_derivatives_for_mapping(
            mapping=self.mapping,
            coordinate_system=self.coordinate_system,
            spatial_dim=spatial_dim,
            other_dims=other_dims,
            central=central,
            out=out,
        )

    def estimate_spatial_jacobian_matrices(
        self,
        other_dims: str = "average",
        central: bool = False,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        """Estimate spatial Jacobian matrices over the mapping at the grid locations

        Args:
            other_dims: How to handle the other dimensions, see
                finite_difference.estimate_spatial_jacobian_matrices for more details
            central: Whether to use central differences
            out: Output tensor to store the result
        """
        return estimate_spatial_jacobian_matrices_for_mapping(
            mapping=self.mapping,
            coordinate_system=self.coordinate_system,
            other_dims=other_dims,
            central=central,
            out=out,
        )


class SamplableDeformationMapping(BaseSamplableMapping):
    """Wrapper for composable mapping deformation bundled together with a coordinate system

    Arguments:
        mapping: Composable mapping to be wrapped
        coordinate_system: Coordinate system to use for sampling and resampling
        grid_mapping_args: Grid mapping arguments to use for resampling
        resample_as: How to perform potential resampling, either
            "displacement_field" or "coordinate_field"
    """

    def __init__(
        self,
        mapping: IComposableMapping,
        coordinate_system: IVoxelCoordinateSystem,
        grid_mapping_args: GridMappingArgs,
        resample_as: str = "displacement_field",
    ):
        super().__init__(
            mapping=mapping,
            coordinate_system=coordinate_system,
            grid_mapping_args=grid_mapping_args,
        )
        self.resample_as = resample_as

    def as_displacement_field(self, in_voxel_coordinates: bool = True) -> IMaskedTensor:
        """Return the mapping as displacement field"""
        coordinates = self.sample()
        if not in_voxel_coordinates:
            return coordinates.modify_values(
                coordinates.generate_values() - self.coordinate_system.grid.generate_values()
            )
        voxel_coordinates = self.coordinate_system.to_voxel_coordinates(coordinates)
        return voxel_coordinates.modify_values(
            voxel_coordinates.generate_values()
            - self.coordinate_system.voxel_grid.generate_values()
        )

    def resample(self) -> "SamplableDeformationMapping":
        """Resample the mapping"""
        if self.resample_as == "displacement_field":
            data = self.as_displacement_field(in_voxel_coordinates=True)
        elif self.resample_as == "coordinate_field":
            data = self.coordinate_system.to_voxel_coordinates(self.sample())
        else:
            raise ValueError(f"Invalid resample_as value: {self.resample_as}")
        return SamplableDeformationMapping(
            mapping=create_deformation_from_voxel_data(
                data=data,
                grid_mapping_args=self.grid_mapping_args,
                coordinate_system=self.coordinate_system,
                data_format=self.resample_as,
            ),
            coordinate_system=self.coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
            resample_as=self.resample_as,
        )

    def _simplified_modified_copy(
        self, mapping: IComposableMapping, coordinate_system: IVoxelCoordinateSystem
    ) -> "SamplableDeformationMapping":
        return SamplableDeformationMapping(
            mapping=mapping,
            coordinate_system=coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
            resample_as=self.resample_as,
        )

    def __repr__(self) -> str:
        return (
            f"SamplableVolumeMapping(mapping={self.mapping}, "
            f"coordinate_system={self.coordinate_system}, "
            f"grid_mapping_args={self.grid_mapping_args})"
        )

    def visualize(
        self,
        batch_index: int = 0,
        figure_height: int = 5,
        emphasize_every_nth_line: Optional[Tuple[int, int]] = None,
    ) -> Figure:
        """Visualize the mapping as a deformation

        If there are more than two dimension, central slices are shown for each pair of dimensions.

        Args:
            batch_index: Index of the batch element to visualize
            figure_height: Height of the figure
            emphasize_every_nth_line: Tuple of two integers, the first one is the number of lines
                to emphasize, the second one is the offset
        """
        transformed_grid = self.mapping(self.coordinate_system.grid).generate_values()[batch_index]
        n_dims = len(self.coordinate_system.grid.spatial_shape)
        if n_dims > 1:
            grids = []
            dimension_pairs = list(combinations(range(n_dims), 2))
            for dimension_pair in dimension_pairs:
                other_dims = [dim for dim in range(n_dims) if dim not in dimension_pair]
                transformed_grid_2d = transformed_grid[list(dimension_pair)]
                for other_dim in reversed(other_dims):
                    transformed_grid_2d = transformed_grid_2d.movedim(-n_dims + other_dim, 0)
                    transformed_grid_2d = transformed_grid_2d[transformed_grid_2d.size(0) // 2]
                assert transformed_grid_2d.ndim == 3
                assert transformed_grid_2d.size(0) == 2
                grids.append(transformed_grid_2d)
        else:
            raise NotImplementedError("Visualization of 1D deformation is not supported")

        def dimension_to_letter(dim: int) -> str:
            if n_dims <= 3:
                return "xyz"[dim]
            return f"dim_{dim}"

        def get_kwargs(index: int) -> Mapping[str, Any]:
            if emphasize_every_nth_line is None:
                return {}
            if index + emphasize_every_nth_line[1] % emphasize_every_nth_line[0] == 0:
                return {"alpha": 0.6, "linewidth": 2.0}
            return {"alpha": 0.2, "linewidth": 1.0}

        figure, axes = subplots(
            1, len(grids), figsize=(figure_height * len(grids), figure_height), squeeze=False
        )

        for axis, grid, (dim_1, dim_2) in zip(axes.flatten(), grids, dimension_pairs):
            axis.axis("equal")
            axis.set_xlabel(dimension_to_letter(dim_1))
            axis.set_ylabel(dimension_to_letter(dim_2))
            for row_index in range(grid.size(1)):
                axis.plot(
                    grid[0, row_index, :],
                    grid[1, row_index, :],
                    color="gray",
                    **get_kwargs(row_index),
                )
            for col_index in range(grid.size(2)):
                axis.plot(
                    grid[0, :, col_index],
                    grid[1, :, col_index],
                    color="gray",
                    **get_kwargs(col_index),
                )

        return figure


class SamplableVolumeMapping(BaseSamplableMapping):
    """Wrapper for composable mapping volume bundled together with a coordinate system"""

    def resample(self) -> "SamplableVolumeMapping":
        """Resample the mapping"""
        sampled = self.sample()
        resampled_volume = GridVolume(
            data=self.mapping(self.coordinate_system.grid),
            grid_mapping_args=self.grid_mapping_args,
            n_channel_dims=len(sampled.channels_shape),
        )
        return SamplableVolumeMapping(
            mapping=resampled_volume,
            coordinate_system=self.coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
        )

    def _simple_modified_copy(
        self, mapping: GridVolume, coordinate_system: IVoxelCoordinateSystem
    ) -> "SamplableVolumeMapping":
        return SamplableVolumeMapping(
            mapping=mapping,
            coordinate_system=coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
        )

    def __repr__(self) -> str:
        return (
            f"SamplableVolumeMapping(mapping={self.mapping}, "
            f"coordinate_system={self.coordinate_system}, "
            f"grid_mapping_args={self.grid_mapping_args})"
        )
