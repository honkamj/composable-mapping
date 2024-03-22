""""Composable mapping bundled together with a coordinate system"""

from typing import Mapping, Optional, Union

from torch import Tensor

from composable_mapping.finite_difference import (
    estimate_spatial_derivatives_for_mapping,
    estimate_spatial_jacobian_matrices_for_mapping,
)

from .affine import ComposableAffine
from .base import BaseTensorLikeWrapper
from .grid_mapping import (
    GridCoordinateMapping,
    GridMappingArgs,
    GridVolume,
    as_displacement_field,
)
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
            displacement_field, mask = as_displacement_field(
                self.mapping, coordinate_system=self.coordinate_system, generate_missing_mask=False
            )
            resampled_composable_mapping: IComposableMapping = GridCoordinateMapping(
                displacement_field=displacement_field,
                grid_mapping_args=self.grid_mapping_args,
                mask=mask,
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

    def sample(self) -> IMaskedTensor:
        """Sample the mapping"""
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
