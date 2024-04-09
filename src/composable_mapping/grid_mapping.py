"""Grid based mappings"""

from typing import Callable, Literal, Mapping, Optional, Tuple, Union, overload

from deformation_inversion_layer import (
    DeformationInversionArguments,
    fixed_point_invert_deformation,
)
from deformation_inversion_layer.interface import Interpolator
from torch import Tensor

from .base import BaseComposableMapping
from .dense_deformation import compute_fov_mask_at_voxel_coordinates
from .interface import (
    IComposableMapping,
    IMaskedTensor,
    ITensorLike,
    IVoxelCoordinateSystem,
)
from .masked_tensor import MaskedTensor
from .util import combine_optional_masks


class GridMappingArgs:
    """Represents arguments for creating grid volume

    Arguments:
        interpolator: Interpolator which interpolates the volume at voxel coordinates
        mask_interpolator: Interpolator which interpolates the mask at voxel coordinates,
            defaults to interpolator
        mask_outside_fov: Whether to update interpolation locations outside field of view
            to the mask
        mask_threshold: All values under threshold are set to zero and above it to one,
            if None, no thresholding will be done
    """

    def __init__(
        self,
        interpolator: Interpolator,
        mask_interpolator: Optional[Interpolator] = None,
        mask_outside_fov: bool = True,
        mask_threshold: Optional[float] = 1.0 - 1e-5,
    ) -> None:
        self.interpolator = interpolator
        self.mask_interpolator = interpolator if mask_interpolator is None else mask_interpolator
        self.mask_outside_fov = mask_outside_fov
        self.mask_threshold = mask_threshold

    def __repr__(self) -> str:
        return (
            f"GridMappingArgs(interpolator={self.interpolator}, "
            f"mask_interpolator={self.mask_interpolator}, "
            f"mask_outside_fov={self.mask_outside_fov}, "
            f"mask_threshold={self.mask_threshold})"
        )


class GridVolume(BaseComposableMapping):
    """Continuously defined volume on voxel coordinates based on
    and interpolation/extrapolation method

    Arguments:
        data: Tensor with shape (batch_size, *channel_dims, dim_1, ..., dim_{n_dims})
        grid_mapping_args: Additional grid based mapping args
        mask: Mask defining invalid regions,
            Tensor with shape (batch_size, *(1,) * len(channel_dims), dim_1, ..., dim_{n_dims})
        n_channel_dims: Number of channel dimensions in data
    """

    def __init__(
        self,
        data: Tensor,
        grid_mapping_args: GridMappingArgs,
        n_channel_dims: int = 1,
        mask: Optional[Tensor] = None,
    ) -> None:
        self._data = data
        self._mask = mask
        self._grid_mapping_args = grid_mapping_args
        self._n_channel_dims = n_channel_dims
        self._volume_shape = data.shape[n_channel_dims + 1 :]
        self._n_dims = len(self._volume_shape)
        if mask is not None and mask.device != data.device:
            raise RuntimeError("Devices do not match")

    def _get_tensors(self) -> Mapping[str, Tensor]:
        tensors = {
            "data": self._data,
        }
        if self._mask is not None:
            tensors["mask"] = self._mask
        return tensors

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "GridVolume":
        return GridVolume(
            data=tensors["data"],
            grid_mapping_args=self._grid_mapping_args,
            n_channel_dims=self._n_channel_dims,
            mask=tensors.get("mask"),
        )

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return self._evaluate(
            voxel_coordinates_generator=masked_coordinates.generate_values,
            coordinate_mask=masked_coordinates.generate_mask(generate_missing_mask=False),
            coordinates_as_slice=masked_coordinates.as_slice(self._volume_shape),
        )

    def _evaluate(
        self,
        voxel_coordinates_generator: Callable[[], Tensor],
        coordinate_mask: Optional[Tensor],
        coordinates_as_slice: Optional[Tuple[Union["ellipsis", slice], ...]],
    ) -> MaskedTensor:
        if coordinates_as_slice is None:
            voxel_coordinates = voxel_coordinates_generator()
            values = self._grid_mapping_args.interpolator(self._data, voxel_coordinates)
            mask = self._evaluate_mask(voxel_coordinates)
        else:
            values = self._data[coordinates_as_slice]
            mask = self._mask[coordinates_as_slice] if self._mask is not None else None
        mask = combine_optional_masks([mask, coordinate_mask])
        return MaskedTensor(values, mask, self._n_channel_dims)

    def _evaluate_mask(self, voxel_coordinates: Tensor) -> Optional[Tensor]:
        if self._mask is None:
            interpolated_mask: Optional[Tensor] = None
        else:
            interpolated_mask = self._grid_mapping_args.mask_interpolator(
                self._mask, voxel_coordinates
            )
            interpolated_mask = self._threshold_mask(interpolated_mask)
        fov_mask = (
            compute_fov_mask_at_voxel_coordinates(
                voxel_coordinates,
                volume_shape=self._volume_shape,
                dtype=voxel_coordinates.dtype,
            )
            if self._grid_mapping_args.mask_outside_fov
            else None
        )
        mask = combine_optional_masks([interpolated_mask, fov_mask])
        return mask

    def _threshold_mask(self, mask: Tensor) -> Tensor:
        if self._grid_mapping_args.mask_threshold is not None:
            return (mask < self._grid_mapping_args.mask_threshold).logical_not().type(mask.dtype)
        return mask

    def invert(self, **kwargs) -> IComposableMapping:
        raise NotImplementedError("No inversion implemented for grid volumes")

    def __repr__(self) -> str:
        return (
            f"GridVolume(data={self._data}, grid_mapping_args={self._grid_mapping_args}, "
            f"mask={self._mask}, n_channel_dims={self._n_channel_dims})"
        )


class GridCoordinateMapping(GridVolume):
    """Continuously defined mapping based on regular grid samples

    Arguments:
        displacement_field: Displacement field in voxel coordinates, Tensor with shape
            (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        grid_mapping_args: Additional grid based mapping args
        mask: Mask defining invalid regions,
            Tensor with shape (batch_size, *(1,) * len(channel_dims), dim_1, ..., dim_{n_dims})
    """

    def __init__(
        self,
        displacement_field: Tensor,
        grid_mapping_args: GridMappingArgs,
        mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            data=displacement_field,
            grid_mapping_args=grid_mapping_args,
            mask=mask,
            n_channel_dims=1,
        )

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "GridCoordinateMapping":
        return GridCoordinateMapping(
            displacement_field=tensors["data"],
            grid_mapping_args=self._grid_mapping_args,
            mask=tensors.get("mask"),
        )

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        voxel_coordinates, coordinate_mask = masked_coordinates.generate(
            generate_missing_mask=False
        )
        displacement_field_values = super()._evaluate(
            voxel_coordinates_generator=lambda: voxel_coordinates,
            coordinate_mask=coordinate_mask,
            coordinates_as_slice=masked_coordinates.as_slice(self._volume_shape),
        )
        return MaskedTensor(
            values=voxel_coordinates + displacement_field_values.generate_values(),
            mask=displacement_field_values.generate_mask(generate_missing_mask=False),
        )

    @overload
    def invert(
        self: "GridCoordinateMapping", **inversion_parameters
    ) -> "_GridCoordinateMappingInverse": ...

    # overload is required for the subclass
    @overload
    def invert(  # type: ignore
        self: "_GridCoordinateMappingInverse", **inversion_parameters
    ) -> "GridCoordinateMapping": ...

    def invert(self, **inversion_parameters):
        """Fixed point invert displacement field

        inversion_parameters:
            fixed_point_inversion_arguments (Mapping[str, Any]): Arguments for
                fixed point inversion
        """
        fixed_point_inversion_arguments = inversion_parameters.get(
            "fixed_point_inversion_arguments", {}
        )
        deformation_inversion_arguments = DeformationInversionArguments(
            interpolator=self._grid_mapping_args.interpolator,
            forward_solver=fixed_point_inversion_arguments.get("forward_solver"),
            backward_solver=fixed_point_inversion_arguments.get("backward_solver"),
            forward_dtype=fixed_point_inversion_arguments.get("forward_dtype"),
            backward_dtype=fixed_point_inversion_arguments.get("backward_dtype"),
        )
        return _GridCoordinateMappingInverse(
            displacement_field=self._data,
            grid_mapping_args=self._grid_mapping_args,
            inversion_arguments=deformation_inversion_arguments,
            mask=self._mask,
        )

    def __repr__(self) -> str:
        return (
            f"GridCoordinateMapping(displacement_field={self._data}, "
            f"grid_mapping_args={self._grid_mapping_args}, mask={self._mask})"
        )


class _GridCoordinateMappingInverse(GridCoordinateMapping):
    """Inverse of a continuously defined mapping based on regular grid samples

    The inverse is computed using a fixed point iteration

    Arguments:
        displacement_field: Displacement field in voxel coordinates, Tensor with shape
            (batch_size, n_dims, dim_1, ..., dim_{n_dims})
        grid_mapping_args: Additional grid based mapping args
        inversion_arguments: Arguments for fixed point inversion
        mask: Mask defining invalid regions,
            Tensor with shape (batch_size, *(1,) * len(channel_dims), dim_1, ..., dim_{n_dims})
    """

    def __init__(
        self,
        displacement_field: Tensor,
        grid_mapping_args: GridMappingArgs,
        inversion_arguments: DeformationInversionArguments,
        mask: Optional[Tensor] = None,
    ) -> None:
        super().__init__(
            displacement_field=displacement_field,
            grid_mapping_args=grid_mapping_args,
            mask=mask,
        )
        self._inversion_arguments = inversion_arguments

    def _evaluate(
        self,
        voxel_coordinates_generator: Callable[[], Tensor],
        coordinate_mask: Optional[Tensor],
        coordinates_as_slice: Optional[Tuple[Union["ellipsis", slice], ...]],
    ) -> MaskedTensor:
        voxel_coordinates = voxel_coordinates_generator()
        inverted_values = fixed_point_invert_deformation(
            displacement_field=self._data,
            arguments=self._inversion_arguments,
            initial_guess=(
                None if coordinates_as_slice is None else -self._data[coordinates_as_slice]
            ),
            coordinates=voxel_coordinates,
        )
        mask = self._evaluate_mask(voxel_coordinates + inverted_values)
        mask = combine_optional_masks([mask, coordinate_mask])
        return MaskedTensor(inverted_values, mask, self._n_channel_dims)

    def invert(self, **_inversion_parameters):
        return GridCoordinateMapping(
            displacement_field=self._data,
            grid_mapping_args=self._grid_mapping_args,
            mask=self._mask,
        )

    def __repr__(self) -> str:
        return (
            f"_GridCoordinateMappingInverse(displacement_field={self._data}, "
            f"grid_mapping_args={self._grid_mapping_args}, mask={self._mask})"
        )


@overload
def as_displacement_field(
    mapping: IComposableMapping,
    coordinate_system: IVoxelCoordinateSystem,
    generate_missing_mask: Literal[True] = ...,
) -> Tuple[Tensor, Tensor]: ...


@overload
def as_displacement_field(
    mapping: IComposableMapping,
    coordinate_system: IVoxelCoordinateSystem,
    generate_missing_mask: Literal[False],
) -> Tuple[Tensor, Optional[Tensor]]: ...


@overload
def as_displacement_field(
    mapping: IComposableMapping,
    coordinate_system: IVoxelCoordinateSystem,
    generate_missing_mask: bool,
) -> Tuple[Tensor, Optional[Tensor]]: ...


def as_displacement_field(
    mapping: IComposableMapping,
    coordinate_system: IVoxelCoordinateSystem,
    generate_missing_mask: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Extract displacement field from a mapping"""
    voxel_coordinate_mapping = coordinate_system.to_voxel_coordinates(
        mapping(coordinate_system.grid)
    )
    voxel_coordinate_mapping_values, voxel_coordinate_mapping_mask = (
        voxel_coordinate_mapping.generate(generate_missing_mask=generate_missing_mask)
    )
    displacement_field = (
        voxel_coordinate_mapping_values - coordinate_system.voxel_grid.generate_values()
    )
    return displacement_field, voxel_coordinate_mapping_mask
