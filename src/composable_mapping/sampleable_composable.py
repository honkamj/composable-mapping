""""Composable mapping bundled together with a coordinate system"""

from itertools import combinations
from typing import Any, Mapping, Optional, Tuple, Union

from matplotlib.figure import Figure  # type: ignore
from matplotlib.pyplot import subplots  # type: ignore
from torch import Tensor

from composable_mapping.finite_difference import (
    estimate_spatial_derivatives_for_mapping,
    estimate_spatial_jacobian_matrices_for_mapping,
)
from composable_mapping.masked_tensor import MaskedTensor

from .affine import ComposableAffine
from .base import BaseTensorLikeWrapper
from .grid_mapping import GridCoordinateMapping, GridMappingArgs, GridVolume
from .interface import (
    IAffineTransformation,
    IComposableMapping,
    IMaskedTensor,
    ITensorLike,
    IVoxelCoordinateSystem,
)


class SamplableComposable(BaseTensorLikeWrapper):
    """Wrapper for composable mapping bundled together with a coordinate system

    Arguments:
        mapping: Composable mapping to be wrapped
        coordinate_system: Coordinate system to use for sampling
        grid_mapping_args: Grid mapping arguments to use for resampling
        is_deformation: Whether the mapping is a deformation, defines how to resample
            (as a displacement field or a regular volume)
    """

    def __init__(
        self,
        mapping: IComposableMapping,
        coordinate_system: IVoxelCoordinateSystem,
        grid_mapping_args: GridMappingArgs,
        is_deformation: bool = False,
    ):
        self.mapping = mapping
        self.coordinate_system = coordinate_system
        self.is_deformation = is_deformation
        self.grid_mapping_args = grid_mapping_args

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return self.mapping(masked_coordinates)

    def invert(self, **inversion_parameters) -> "SamplableComposable":
        """Invert the mapping"""
        return SamplableComposable(
            mapping=self.mapping.invert(**inversion_parameters),
            coordinate_system=self.coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
            is_deformation=self.is_deformation,
        )

    def resample(self) -> "SamplableComposable":
        """Resample the mapping"""
        if self.is_deformation:
            displacement_field = self.sample(as_displacement_field=True)
            resampled_composable_mapping: IComposableMapping = GridCoordinateMapping(
                displacement_field=displacement_field.generate_values(),
                grid_mapping_args=self.grid_mapping_args,
                mask=displacement_field.generate_mask(generate_missing_mask=False),
            )
        else:
            masked_data = self.mapping(self.coordinate_system.grid)
            data, mask = masked_data.generate(generate_missing_mask=False)
            resampled_composable_mapping = GridVolume(
                data=data,
                grid_mapping_args=self.grid_mapping_args,
                n_channel_dims=len(masked_data.channels_shape),
                mask=mask,
            )
        return SamplableComposable(
            mapping=resampled_composable_mapping,
            coordinate_system=self.coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
            is_deformation=True,
        )

    def sample(self, as_displacement_field: bool = False) -> IMaskedTensor:
        """Sample the mapping"""
        if as_displacement_field:
            values, mask = self.coordinate_system.to_voxel_coordinates(
                self.mapping(self.coordinate_system.grid)
            ).generate()
            displacement_field = values - self.coordinate_system.voxel_grid.generate_values()
            return MaskedTensor(displacement_field, mask)
        return self.mapping(self.coordinate_system.grid)

    def __matmul__(
        self, right_mapping: Union["SamplableComposable", IComposableMapping, IAffineTransformation]
    ) -> "SamplableComposable":
        if isinstance(right_mapping, IAffineTransformation):
            right_composable_mapping: IComposableMapping = ComposableAffine(right_mapping)
            coordinate_system = self.coordinate_system
            is_deformation = True
        elif isinstance(right_mapping, IComposableMapping):
            right_composable_mapping = right_mapping
            coordinate_system = self.coordinate_system
            is_deformation = self.is_deformation
        else:
            right_composable_mapping = right_mapping.mapping
            coordinate_system = right_mapping.coordinate_system
            is_deformation = right_mapping.is_deformation
        return SamplableComposable(
            mapping=self.mapping.compose(right_composable_mapping),
            coordinate_system=coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
            is_deformation=self.is_deformation and is_deformation,
        )

    def __rmatmul__(
        self, left_mapping: Union["SamplableComposable", IAffineTransformation]
    ) -> "SamplableComposable":
        if isinstance(left_mapping, SamplableComposable):
            return left_mapping.__matmul__(self)
        return SamplableComposable(
            mapping=ComposableAffine(left_mapping).compose(self.mapping),
            coordinate_system=self.coordinate_system,
            grid_mapping_args=self.grid_mapping_args,
            is_deformation=self.is_deformation,
        )

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"mapping": self.mapping, "coordinate_system": self.coordinate_system}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "SamplableComposable":
        if not isinstance(children["mapping"], IComposableMapping) or not isinstance(
            children["coordinate_system"], IVoxelCoordinateSystem
        ):
            raise ValueError("Invalid children for samplable composable")
        return SamplableComposable(
            mapping=children["mapping"],
            coordinate_system=children["coordinate_system"],
            grid_mapping_args=self.grid_mapping_args,
            is_deformation=self.is_deformation,
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

    def __repr__(self) -> str:
        return (
            f"SamplableComposable(mapping={self.mapping}, "
            f"coordinate_system={self.coordinate_system}, "
            f"grid_mapping_args={self.grid_mapping_args}, "
            f"is_deformation={self.is_deformation})"
        )

    def visualize_as_deformation(
        self,
        batch_index: int = 0,
        figure_height: int = 5,
        emphasize_every_nth_line: Optional[Tuple[int, int]] = None,
    ) -> Figure:
        """Visualize the mapping as a deformation

        If there are more than two dimension, central slices are shown for each pair of dimensions.

        Args
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
