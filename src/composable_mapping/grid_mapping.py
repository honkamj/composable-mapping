"""Grid based mappings"""

from contextlib import ContextDecorator
from threading import local
from typing import Callable, Mapping, Optional, Tuple, Union

from deformation_inversion_layer import (
    DeformationInversionArguments,
    fixed_point_invert_deformation,
)
from deformation_inversion_layer.interface import Interpolator
from deformation_inversion_layer.interpolator import LinearInterpolator
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


class InterpolationArgs:
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
            f"InterpolationArgs(interpolator={self.interpolator}, "
            f"mask_interpolator={self.mask_interpolator}, "
            f"mask_outside_fov={self.mask_outside_fov}, "
            f"mask_threshold={self.mask_threshold})"
        )


class GridVolume(BaseComposableMapping):
    """Continuously defined mapping based on regular grid samples

    Arguments:
        data: Regular grid of values defining the deformation, with shape
            (batch_size, n_dims, dim_1, ..., dim_{n_dims}). The grid should be
            in voxel coordinates.
        interpolation_args: Arguments defining interpolation and extrapolation behavior
        mask: Mask defining invalid regions,
            Tensor with shape (batch_size, *(1,) * len(channel_dims), dim_1, ..., dim_{n_dims})
        n_channel_dims: Number of channel dimensions in data
    """

    def __init__(
        self,
        data: IMaskedTensor,
        interpolation_args: InterpolationArgs,
        n_channel_dims: int = 1,
    ) -> None:
        self._data = data
        self._interpolation_args = interpolation_args
        self._n_channel_dims = n_channel_dims
        self._volume_shape = data.shape[n_channel_dims + 1 :]
        self._n_dims = len(self._volume_shape)

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {
            "data": self._data,
        }

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "GridVolume":
        if not isinstance(children["data"], IMaskedTensor):
            raise ValueError("Invalid children for samplable composable")
        return GridVolume(
            data=children["data"],
            interpolation_args=self._interpolation_args,
            n_channel_dims=self._n_channel_dims,
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
        data, data_mask = self._data.generate(generate_missing_mask=False)
        if coordinates_as_slice is None:
            voxel_coordinates = voxel_coordinates_generator()
            values = self._interpolation_args.interpolator(data, voxel_coordinates)
            mask = self._evaluate_mask(voxel_coordinates, data_mask)
        else:
            values = data[coordinates_as_slice]
            mask = data_mask[coordinates_as_slice] if data_mask is not None else None
        mask = combine_optional_masks([mask, coordinate_mask])
        return MaskedTensor(values, mask, self._n_channel_dims)

    def _evaluate_mask(
        self, voxel_coordinates: Tensor, data_mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        if data_mask is None:
            interpolated_mask: Optional[Tensor] = None
        else:
            interpolated_mask = self._interpolation_args.mask_interpolator(
                data_mask, voxel_coordinates
            )
            interpolated_mask = self._threshold_mask(interpolated_mask)
        fov_mask = (
            compute_fov_mask_at_voxel_coordinates(
                voxel_coordinates,
                volume_shape=self._volume_shape,
                dtype=voxel_coordinates.dtype,
            )
            if self._interpolation_args.mask_outside_fov
            else None
        )
        mask = combine_optional_masks([interpolated_mask, fov_mask])
        return mask

    def _threshold_mask(self, mask: Tensor) -> Tensor:
        if self._interpolation_args.mask_threshold is not None:
            return (mask >= self._interpolation_args.mask_threshold).type(mask.dtype)
        return mask

    def invert(self, **inversion_parameters) -> IComposableMapping:
        raise NotImplementedError("No inversion implemented for grid volumes")

    def __repr__(self) -> str:
        return (
            f"GridVolume(data={self._data}, interpolation_args={self._interpolation_args}, "
            f"n_channel_dims={self._n_channel_dims})"
        )


class _BaseGridDeformation(GridVolume):
    """Base class for continuously defined mapping based on regular grid samples
    whose values have the same dimensionality as the spatial dimensions"""

    def __init__(
        self,
        data: IMaskedTensor,
        interpolation_args: InterpolationArgs,
        data_format: str = "displacement_field",
    ) -> None:
        if data.shape[1] != len(data.shape) - 2:
            raise ValueError("Data should have the same dimensionality as the spatial dimensions")
        super().__init__(
            data=data,
            interpolation_args=interpolation_args,
            n_channel_dims=1,
        )
        self._data_format = data_format

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        if self._data_format == "displacement_field":
            voxel_coordinates, coordinate_mask = masked_coordinates.generate(
                generate_missing_mask=False
            )

            def voxel_coordinates_generator():
                return voxel_coordinates

        elif self._data_format == "coordinate_field":
            voxel_coordinates_generator = masked_coordinates.generate_values
            coordinate_mask = masked_coordinates.generate_mask(generate_missing_mask=False)
        else:
            raise ValueError(f"Invalid data format: {self._data_format}")
        values = self._evaluate(
            voxel_coordinates_generator=voxel_coordinates_generator,
            coordinate_mask=coordinate_mask,
            coordinates_as_slice=masked_coordinates.as_slice(self._volume_shape),
        )
        if self._data_format == "displacement_field":
            values = values.modify_values(values=values.generate_values() + voxel_coordinates)
        return values

    def __repr__(self) -> str:
        return (
            f"GridCoordinateMapping(displacement_field={self._data}, "
            f"interpolation_args={self._interpolation_args})"
        )


class GridDeformation(_BaseGridDeformation):
    """Continuously defined mapping based on regular grid samples whose values
    have the same dimensionality as the spatial dimensions

    Arguments:
        data: (batch_size, n_dims, dim_1, ..., dim_{n_dims}). The values are
            either directly the coordinates or a displacement field. The values
            should be given in voxel coordinates, and the grid is also assumed
            to be in voxel coordinates.
        interpolation_args: Arguments defining interpolation and extrapolation behavior
        data_format: Format of the provided data, either "displacement_field" or "coordinate_field"
    """

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "GridDeformation":
        if not isinstance(children["data"], IMaskedTensor):
            raise ValueError("Invalid children for samplable composable")
        return GridDeformation(
            data=children["data"],
            interpolation_args=self._interpolation_args,
            data_format=self._data_format,
        )

    def invert(self, **inversion_parameters) -> "_GridDeformationInverse":
        """Invert the deformation

        inversion_parameters:
            fixed_point_inversion_arguments (Mapping[str, Any]): Arguments for
                fixed point inversion
        """
        fixed_point_inversion_arguments = inversion_parameters.get(
            "fixed_point_inversion_arguments", {}
        )
        deformation_inversion_arguments = DeformationInversionArguments(
            interpolator=self._interpolation_args.interpolator,
            forward_solver=fixed_point_inversion_arguments.get("forward_solver"),
            backward_solver=fixed_point_inversion_arguments.get("backward_solver"),
            forward_dtype=fixed_point_inversion_arguments.get("forward_dtype"),
            backward_dtype=fixed_point_inversion_arguments.get("backward_dtype"),
        )
        return _GridDeformationInverse(
            inverted_data=self._data,
            interpolation_args=self._interpolation_args,
            inversion_arguments=deformation_inversion_arguments,
            data_format=self._data_format,
        )


class _GridDeformationInverse(_BaseGridDeformation):
    """Inverse of a GridDeformation

    The inverse is computed using a fixed point iteration
    """

    def __init__(
        self,
        inverted_data: IMaskedTensor,
        interpolation_args: InterpolationArgs,
        inversion_arguments: DeformationInversionArguments,
        data_format: str,
    ) -> None:
        super().__init__(
            data=inverted_data,
            interpolation_args=interpolation_args,
            data_format="displacement_field",
        )
        self._inversion_arguments = inversion_arguments
        self._inverted_data_format = data_format

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_GridDeformationInverse":
        if not isinstance(children["data"], IMaskedTensor):
            raise ValueError("Invalid children for samplable composable")
        return _GridDeformationInverse(
            inverted_data=children["data"],
            interpolation_args=self._interpolation_args,
            inversion_arguments=self._inversion_arguments,
            data_format=self._inverted_data_format,
        )

    def _evaluate(
        self,
        voxel_coordinates_generator: Callable[[], Tensor],
        coordinate_mask: Optional[Tensor],
        coordinates_as_slice: Optional[Tuple[Union["ellipsis", slice], ...]],
    ) -> MaskedTensor:
        data, data_mask = self._data.generate(generate_missing_mask=False)
        voxel_coordinates = voxel_coordinates_generator()
        inverted_values = fixed_point_invert_deformation(
            displacement_field=(
                data
                if self._inverted_data_format == "displacement_field"
                else data - voxel_coordinates
            ),
            arguments=self._inversion_arguments,
            initial_guess=(None if coordinates_as_slice is None else -data[coordinates_as_slice]),
            coordinates=voxel_coordinates,
        )
        mask = self._evaluate_mask(voxel_coordinates + inverted_values, data_mask)
        mask = combine_optional_masks([mask, coordinate_mask])
        return MaskedTensor(inverted_values, mask, self._n_channel_dims)

    def invert(self, **inversion_parameters) -> GridDeformation:
        return GridDeformation(
            data=self._data,
            interpolation_args=self._interpolation_args,
            data_format=self._inverted_data_format,
        )

    def __repr__(self) -> str:
        return (
            f"_GridCoordinateMappingInverse(displacement_field={self._data}, "
            f"interpolation_args={self._interpolation_args})"
        )


def create_volume(
    data: IMaskedTensor,
    interpolation_args: InterpolationArgs,
    coordinate_system: IVoxelCoordinateSystem,
    n_channel_dims: int = 1,
) -> IComposableMapping:
    """Create volume based on grid samples"""
    grid_volume = GridVolume(
        data=data,
        interpolation_args=interpolation_args,
        n_channel_dims=n_channel_dims,
    )
    return grid_volume.compose(coordinate_system.to_voxel_coordinates)


def create_deformation_from_voxel_data(
    data: IMaskedTensor,
    interpolation_args: InterpolationArgs,
    coordinate_system: IVoxelCoordinateSystem,
    data_format: str = "displacement_field",
) -> IComposableMapping:
    """Create deformation mapping based on grid samples in given coordinate system
    with data given in voxel coordinates

    Args:
        data_in_voxel_coordinates: Regular grid of values defining the deformation,
            Tensor with shape
            (batch_size, n_dims, dim_1, ..., dim_{n_dims}). The values are
            either directly the coordinates or a displacement field. The values
            should be given in voxel coordinates, and the grid is also assumed
            to be in voxel coordinates.
        interpolation_args: Arguments defining interpolation and extrapolation behavior
        coordinate_system: Coordinate system defining transformations between voxel and
            world coordinates
        data_format: Format of the provided data, either "displacement_field" or "coordinate_field"
    """
    grid_volume = GridDeformation(
        data=data,
        interpolation_args=interpolation_args,
        data_format=data_format,
    )
    return coordinate_system.from_voxel_coordinates.compose(grid_volume).compose(
        coordinate_system.to_voxel_coordinates
    )


def create_deformation_from_world_data(
    data: IMaskedTensor,
    interpolation_args: InterpolationArgs,
    coordinate_system: IVoxelCoordinateSystem,
    data_format: str = "displacement_field",
) -> IComposableMapping:
    """Create deformation mapping based on grid samples in given coordinate system
    with data given in world coordinates

    Works by first transforming the data to voxel coordinates and then creating the
    deformation mapping.

    Args:
        data_in_world_coordinates: Regular grid of values defining the deformation,
            Tensor with shape
            (batch_size, n_dims, dim_1, ..., dim_{n_dims}). The values are
            either directly the coordinates or a displacement field. The values
            should be given in world coordinates, and the grid is also assumed
            to be in world coordinates.
        interpolation_args: Arguments defining interpolation and extrapolation behavior
        coordinate_system: Coordinate system defining transformations between voxel and
            world coordinates
        data_format: Format of the provided data, either "displacement_field" or "coordinate_field"
    """
    if data_format == "displacement_field":
        ddf, ddf_mask = data.generate(generate_missing_mask=False)
        coordinates_in_voxel_coordinates, mask_in_voxel_coordinates = (
            coordinate_system.to_voxel_coordinates(
                MaskedTensor(ddf + coordinate_system.grid.generate_values(), mask=ddf_mask)
            ).generate()
        )
        data_in_voxel_coordinates: IMaskedTensor = MaskedTensor(
            coordinates_in_voxel_coordinates - coordinate_system.voxel_grid.generate_values(),
            mask=mask_in_voxel_coordinates,
        )
    elif data_format == "coordinate_field":
        data_in_voxel_coordinates = coordinate_system.to_voxel_coordinates(data).reduce()
    else:
        raise ValueError(f"Invalid data format: {data_format}")
    return create_deformation_from_voxel_data(
        data=data_in_voxel_coordinates,
        interpolation_args=interpolation_args,
        coordinate_system=coordinate_system,
        data_format=data_format,
    )


_DEFAULT_INTERPOLATION_ARGS: Optional[InterpolationArgs] = None
_DEFAULT_INTERPOLATION_ARGS_CONTEXT_STACK = local()
_DEFAULT_INTERPOLATION_ARGS_CONTEXT_STACK.stack = []


def get_default_interpolation_args() -> InterpolationArgs:
    """Get current default interpolation args"""
    if _DEFAULT_INTERPOLATION_ARGS_CONTEXT_STACK.stack:
        return _DEFAULT_INTERPOLATION_ARGS_CONTEXT_STACK.stack[-1]
    if _DEFAULT_INTERPOLATION_ARGS is None:
        return InterpolationArgs(interpolator=LinearInterpolator())
    return _DEFAULT_INTERPOLATION_ARGS


def set_default_interpolation_args(interpolation_args: Optional[InterpolationArgs]) -> None:
    """Set default interpolation args"""
    global _DEFAULT_INTERPOLATION_ARGS  # pylint: disable=global-statement
    _DEFAULT_INTERPOLATION_ARGS = interpolation_args


def clear_default_interpolation_args() -> None:
    """Clear default interpolation args"""
    global _DEFAULT_INTERPOLATION_ARGS  # pylint: disable=global-statement
    _DEFAULT_INTERPOLATION_ARGS = None


def get_interpolation_args(interpolation_args: Optional[InterpolationArgs]) -> InterpolationArgs:
    """Get interpolation args, either from argument or default"""
    return (
        interpolation_args if interpolation_args is not None else get_default_interpolation_args()
    )


class default_interpolation_args(  # this is supposed to appear as function - pylint: disable=invalid-name
    ContextDecorator
):
    """Context manager for setting default interpolation args"""

    def __init__(self, interpolation_args: InterpolationArgs):
        self.interpolation_args = interpolation_args

    def __enter__(self) -> None:
        _DEFAULT_INTERPOLATION_ARGS_CONTEXT_STACK.stack.append(self.interpolation_args)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _DEFAULT_INTERPOLATION_ARGS_CONTEXT_STACK.stack.pop()
