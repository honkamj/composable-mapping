""""Composable mapping bundled together with a coordinate system"""

from itertools import combinations
from typing import Any, Literal, Mapping, Optional, Tuple, TypeVar, Union, overload

from matplotlib.figure import Figure  # type: ignore
from matplotlib.pyplot import subplots  # type: ignore
from torch import Tensor

from .affine import (
    ComposableAffine,
    NotAffineTransformationError,
    as_affine_transformation,
)
from .base import BaseTensorLikeWrapper
from .finite_difference import (
    estimate_spatial_derivatives_for_mapping,
    estimate_spatial_jacobian_matrices_for_mapping,
)
from .grid_mapping import (
    InterpolationArgs,
    create_deformation_from_voxel_data,
    create_volume,
)
from .interface import (
    IAffineTransformation,
    IComposableMapping,
    IMaskedTensor,
    ITensorLike,
    IVoxelCoordinateSystem,
    IVoxelCoordinateSystemFactory,
)

BaseSamplableMappingT = TypeVar("BaseSamplableMappingT", bound="BaseSamplableMapping")


class BaseSamplableMapping(BaseTensorLikeWrapper):
    """Base implementation for composable mappings bundled together with a coordinate system

    Arguments:
        mapping: Composable mapping to be wrapped
        coordinate_system: Coordinate system to use for sampling and resampling
    """

    def __init__(
        self,
        mapping: IComposableMapping,
        coordinate_system: IVoxelCoordinateSystem,
    ):
        self.mapping = mapping
        self.coordinate_system = coordinate_system

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return self.mapping(masked_coordinates)

    def sample(self) -> IMaskedTensor:
        """Sample the mapping"""
        return self.sample_to(self)

    def sample_to(
        self,
        target: Union[
            IMaskedTensor,
            IVoxelCoordinateSystem,
            "BaseSamplableMapping",
            IVoxelCoordinateSystemFactory,
        ],
    ) -> IMaskedTensor:
        """Sample the mapping wtih respect to the target coordinates"""
        if isinstance(target, IVoxelCoordinateSystem):
            target = target.grid
        elif isinstance(target, BaseSamplableMapping):
            target = target.coordinate_system.grid
        elif isinstance(target, IVoxelCoordinateSystemFactory):
            target = target.create_coordinate_system(
                dtype=self.mapping.dtype, device=self.mapping.device
            ).grid
        return self.mapping(target)

    def _obtain_resampling_coordinate_system(
        self,
        target: Union[
            IVoxelCoordinateSystem, "BaseSamplableMapping", IVoxelCoordinateSystemFactory
        ],
    ) -> IVoxelCoordinateSystem:
        if isinstance(target, IVoxelCoordinateSystem):
            return target
        if isinstance(target, BaseSamplableMapping):
            return target.coordinate_system
        if isinstance(target, IVoxelCoordinateSystemFactory):
            return target.create_coordinate_system(
                dtype=self.mapping.dtype, device=self.mapping.device
            )
        raise ValueError(f"Invalid target: {target}")

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"mapping": self.mapping, "coordinate_system": self.coordinate_system}

    def invert(self: BaseSamplableMappingT, **inversion_parameters) -> BaseSamplableMappingT:
        """Invert the mapping"""
        return self.__class__(
            mapping=self.mapping.invert(**inversion_parameters),
            coordinate_system=self.coordinate_system,
        )

    def _simplified_modified_copy(
        self: BaseSamplableMappingT,
        mapping: IComposableMapping,
        coordinate_system: IVoxelCoordinateSystem,
    ) -> BaseSamplableMappingT:
        return self.__class__(mapping=mapping, coordinate_system=coordinate_system)

    def _modified_copy(
        self: BaseSamplableMappingT,
        tensors: Mapping[str, Tensor],
        children: Mapping[str, ITensorLike],
    ) -> BaseSamplableMappingT:
        if not isinstance(children["mapping"], IComposableMapping) or not isinstance(
            children["coordinate_system"], IVoxelCoordinateSystem
        ):
            raise ValueError("Invalid children for samplable mapping")
        return self.__class__(
            mapping=children["mapping"], coordinate_system=children["coordinate_system"]
        )

    def compose(
        self: BaseSamplableMappingT,
        right_mapping: Union[
            "BaseSamplableMapping",
            IComposableMapping,
            IAffineTransformation,
        ],
        *,
        target_coordinate_system: str = "right",
    ) -> BaseSamplableMappingT:
        """Compose the mapping with another mapping with self on the left

        Args:
            right_mapping: Mapping to compose with
            target_coordinate_system: Which coordinate system to use for the resulting mapping,
                either "left" or "right" defining the side of the composition
        """
        if target_coordinate_system not in ["left", "right"]:
            raise ValueError(
                f"Invalid option for target coordinate system: {target_coordinate_system}"
            )
        if isinstance(right_mapping, IAffineTransformation):
            right_composable_mapping: IComposableMapping = ComposableAffine(right_mapping)
            coordinate_system = self.coordinate_system
        elif isinstance(right_mapping, IComposableMapping):
            right_composable_mapping = right_mapping
            coordinate_system = self.coordinate_system
        elif isinstance(right_mapping, BaseSamplableMapping):
            right_composable_mapping = right_mapping.mapping
            coordinate_system = (
                right_mapping.coordinate_system
                if target_coordinate_system == "right"
                else self.coordinate_system
            )
        else:
            return NotImplemented
        return self.__class__(
            mapping=self.mapping.compose(right_composable_mapping),
            coordinate_system=coordinate_system,
        )

    @overload
    def right_compose(
        self: BaseSamplableMappingT,
        left_mapping: IAffineTransformation,
        *,
        target_coordinate_system: str = "right",
    ) -> BaseSamplableMappingT: ...

    @overload
    def right_compose(
        self: BaseSamplableMappingT,
        left_mapping: IComposableMapping,
        *,
        target_coordinate_system: str = "right",
    ) -> BaseSamplableMappingT: ...

    @overload
    def right_compose(
        self,
        left_mapping: BaseSamplableMappingT,
        *,
        target_coordinate_system: str = "right",
    ) -> BaseSamplableMappingT: ...

    def right_compose(
        self,
        left_mapping: Union[
            "BaseSamplableMapping",
            IComposableMapping,
            IAffineTransformation,
        ],
        *,
        target_coordinate_system: str = "right",
    ) -> "BaseSamplableMapping":
        """Compose the mapping with another mapping with self on the right

        Args:
            left_mapping: Mapping to compose with
            target_coordinate_system: Which coordinate system to use for the resulting mapping,
                either "left" or "right" defining the side of the composition
        """
        if target_coordinate_system not in ["left", "right"]:
            raise ValueError(
                f"Invalid option for target coordinate system: {target_coordinate_system}"
            )
        if isinstance(left_mapping, IAffineTransformation):
            left_composable_mapping: IComposableMapping = ComposableAffine(left_mapping)
        elif isinstance(left_mapping, IComposableMapping):
            left_composable_mapping = left_mapping
        elif isinstance(left_mapping, SamplableVolumeMapping) or isinstance(
            left_mapping, SamplableDeformationMapping
        ):
            return left_mapping.compose(self, target_coordinate_system=target_coordinate_system)
        else:
            return NotImplemented
        return self.__class__(
            mapping=left_composable_mapping.compose(self.mapping),
            coordinate_system=self.coordinate_system,
        )

    def __matmul__(
        self: BaseSamplableMappingT,
        right_mapping: Union[
            "BaseSamplableMapping",
            IComposableMapping,
            IAffineTransformation,
        ],
    ) -> BaseSamplableMappingT:
        return self.compose(right_mapping)

    def __rmatmul__(
        self: BaseSamplableMappingT, left_mapping: Union[IAffineTransformation, IComposableMapping]
    ) -> BaseSamplableMappingT:
        return self.right_compose(left_mapping)

    def estimate_spatial_derivatives(
        self,
        spatial_dim: int,
        *,
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
        *,
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
    """

    def sample_to_as_displacement_field(
        self,
        target: Union[
            IVoxelCoordinateSystem,
            "BaseSamplableMapping",
            IVoxelCoordinateSystemFactory,
        ],
        *,
        data_coordinates: str = "voxel",
    ) -> IMaskedTensor:
        """Return the mapping as displacement field with respect to the
        coordinates of the target mapping"""
        if isinstance(target, IVoxelCoordinateSystem):
            coordinate_system = target
        elif isinstance(target, BaseSamplableMapping):
            coordinate_system = target.coordinate_system
        elif isinstance(target, IVoxelCoordinateSystemFactory):
            coordinate_system = target.create_coordinate_system(
                dtype=self.mapping.dtype, device=self.mapping.device
            )
        data = self.sample_to(target)
        if data_coordinates == "world":
            return data.modify_values(
                data.generate_values() - coordinate_system.grid.generate_values()
            )
        if data_coordinates == "voxel":
            voxel_coordinates = coordinate_system.to_voxel_coordinates(data)
            return voxel_coordinates.modify_values(
                voxel_coordinates.generate_values() - coordinate_system.voxel_grid.generate_values()
            )
        raise ValueError(f"Invalid option for data coordinates: {data_coordinates}")

    def sample_as_displacement_field(self, *, data_coordinates: str = "voxel") -> IMaskedTensor:
        """Return the mapping as displacement field"""
        return self.sample_to_as_displacement_field(self, data_coordinates=data_coordinates)

    def resample_to(
        self,
        target: Union[
            IVoxelCoordinateSystem,
            "BaseSamplableMapping",
            IVoxelCoordinateSystemFactory,
        ],
        interpolation_args: Optional[InterpolationArgs] = None,
        *,
        data_format: str = "displacement",
    ) -> "SamplableDeformationMapping":
        """Resample the mapping"""
        coordinate_system = self._obtain_resampling_coordinate_system(target)
        if data_format == "displacement":
            data = self.sample_to_as_displacement_field(coordinate_system, data_coordinates="voxel")
        elif data_format == "coordinate":
            data = coordinate_system.to_voxel_coordinates(self.sample_to(coordinate_system))
        else:
            raise ValueError(f"Invalid data_format option: {data_format}")
        return SamplableDeformationMapping(
            mapping=create_deformation_from_voxel_data(
                data=data,
                interpolation_args=interpolation_args,
                coordinate_system=coordinate_system,
                data_format=data_format,
            ),
            coordinate_system=coordinate_system,
        )

    @overload
    def as_affine(
        self, *, return_none_if_not_affine: Literal[False] = ...
    ) -> IAffineTransformation: ...

    @overload
    def as_affine(self, *, return_none_if_not_affine: bool) -> Optional[IAffineTransformation]: ...

    def as_affine(
        self, *, return_none_if_not_affine: bool = False
    ) -> Optional[IAffineTransformation]:
        """Return the mapping as affine transformation, if possible"""
        try:
            return as_affine_transformation(
                self.mapping, n_dims=len(self.coordinate_system.grid.spatial_shape)
            )
        except NotAffineTransformationError:
            if return_none_if_not_affine:
                return None
            raise

    def __repr__(self) -> str:
        return (
            f"SamplableVolumeMapping(mapping={self.mapping}, "
            f"coordinate_system={self.coordinate_system})"
        )

    @staticmethod
    def _to_numpy(tensor: Tensor) -> Tensor:
        return tensor.detach().cpu().resolve_conj().resolve_neg().numpy()

    def visualize(
        self,
        *,
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
                grids.append(self._to_numpy(transformed_grid_2d))
        else:
            raise NotImplementedError("Visualization of 1D deformation is not supported")

        del transformed_grid

        def dimension_to_letter(dim: int) -> str:
            if n_dims <= 3:
                return "xyz"[dim]
            return f"dim_{dim}"

        def get_kwargs(index: int) -> Mapping[str, Any]:
            if emphasize_every_nth_line is None:
                return {}
            if (index + emphasize_every_nth_line[1]) % emphasize_every_nth_line[0] == 0:
                return {"alpha": 0.6, "linewidth": 2.0}
            return {"alpha": 0.2, "linewidth": 1.0}

        figure, axes = subplots(
            1, len(grids), figsize=(figure_height * len(grids), figure_height), squeeze=False
        )

        for axis, grid, (dim_1, dim_2) in zip(axes.flatten(), grids, dimension_pairs):
            axis.axis("equal")
            axis.set_xlabel(dimension_to_letter(dim_1))
            axis.set_ylabel(dimension_to_letter(dim_2))
            for row_index in range(grid.shape[1]):
                axis.plot(
                    grid[0, row_index, :],
                    grid[1, row_index, :],
                    color="gray",
                    **get_kwargs(row_index),
                )
            for col_index in range(grid.shape[2]):
                axis.plot(
                    grid[0, :, col_index],
                    grid[1, :, col_index],
                    color="gray",
                    **get_kwargs(col_index),
                )

        return figure


class SamplableVolumeMapping(BaseSamplableMapping):
    """Wrapper for composable mapping volume bundled together with a coordinate system"""

    def resample_to(
        self,
        target: Union[
            IVoxelCoordinateSystem,
            "BaseSamplableMapping",
            IVoxelCoordinateSystemFactory,
        ],
        interpolation_args: Optional[InterpolationArgs] = None,
    ) -> "SamplableVolumeMapping":
        """Resample the mapping"""
        coordinate_system = self._obtain_resampling_coordinate_system(target)
        sampled = self.sample_to(coordinate_system)
        return SamplableVolumeMapping(
            mapping=create_volume(
                data=sampled,
                coordinate_system=coordinate_system,
                interpolation_args=interpolation_args,
                n_channel_dims=len(sampled.channels_shape),
            ),
            coordinate_system=coordinate_system,
        )

    def __repr__(self) -> str:
        return (
            f"SamplableVolumeMapping(mapping={self.mapping}, "
            f"coordinate_system={self.coordinate_system})"
        )
