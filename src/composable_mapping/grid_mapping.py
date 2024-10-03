"""Grid based mappings"""

from contextlib import ContextDecorator
from threading import local
from typing import Mapping, Optional

from deformation_inversion_layer import (
    DeformationInversionArguments,
    fixed_point_invert_deformation,
)
from torch import Tensor

from .base import BaseComposableMapping
from .coordinate_system import CoordinateSystem
from .dense_deformation import compute_fov_mask_at_voxel_coordinates
from .interface import IComposableMapping, IInterpolator
from .interpolator import LinearInterpolator
from .mappable_tensor import MappableTensor, PlainTensor
from .tensor_like import ITensorLike
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
        interpolator: Optional[IInterpolator] = None,
        mask_interpolator: Optional[IInterpolator] = None,
        mask_outside_fov: bool = True,
        mask_threshold: Optional[float] = 1.0 - 1e-5,
    ) -> None:
        self.interpolator = LinearInterpolator() if interpolator is None else interpolator
        self.mask_interpolator = (
            self.interpolator if mask_interpolator is None else mask_interpolator
        )
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
        data: MappableTensor,
        interpolation_args: Optional[InterpolationArgs] = None,
    ) -> None:
        self._data = data
        self._interpolation_args = get_interpolation_args(interpolation_args)

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {
            "data": self._data,
        }

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "GridVolume":
        if not isinstance(children["data"], MappableTensor):
            raise ValueError("Invalid children for samplable composable")
        return GridVolume(
            data=children["data"],
            interpolation_args=self._interpolation_args,
        )

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        if coordinates.n_channel_dims != 1:
            raise ValueError("Grid volume can only be sampled with single channel coordinates")
        coordinates_as_slice = coordinates.as_slice(self._data.spatial_shape)
        data, data_mask = self._data.generate(generate_missing_mask=False, cast_mask=False)
        coordinates_mask = coordinates.generate_mask(generate_missing_mask=False)
        if coordinates_as_slice is None:
            voxel_coordinates = coordinates.generate_values()
            values = self._interpolation_args.interpolator(data, voxel_coordinates)
            mask = self._evaluate_mask(voxel_coordinates, data_mask)
        else:
            values = data[coordinates_as_slice]
            mask = data_mask[coordinates_as_slice] if data_mask is not None else None
        mask = combine_optional_masks(
            [mask, coordinates_mask], n_channel_dims=coordinates.n_channel_dims
        )
        return PlainTensor(values, mask, n_channel_dims=len(self._data.channels_shape))

    def _evaluate_mask(
        self, voxel_coordinates: Tensor, data_mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        if data_mask is None:
            interpolated_mask: Optional[Tensor] = None
        else:
            interpolated_mask = self._interpolation_args.mask_interpolator(
                data_mask.to(voxel_coordinates.dtype), voxel_coordinates
            )
            interpolated_mask = self._threshold_mask(interpolated_mask)
        fov_mask = (
            compute_fov_mask_at_voxel_coordinates(
                voxel_coordinates,
                volume_shape=self._data.spatial_shape,
            )
            if self._interpolation_args.mask_outside_fov
            else None
        )
        mask = combine_optional_masks([interpolated_mask, fov_mask])
        return mask

    def _threshold_mask(self, mask: Tensor) -> Tensor:
        if self._interpolation_args.mask_threshold is not None:
            return mask >= self._interpolation_args.mask_threshold
        return mask

    def invert(self, **inversion_parameters) -> IComposableMapping:
        raise NotImplementedError("No inversion implemented for grid volumes")

    def __repr__(self) -> str:
        return f"GridVolume(data={self._data}, interpolation_args={self._interpolation_args}, "


class GridDeformation(GridVolume):
    """Continuously defined mapping based on regular grid samples whose values
    have the same dimensionality as the spatial dimensions

    Arguments:
        data: (batch_size, n_dims, dim_1, ..., dim_{n_dims}). The values are
            either directly the coordinates or a displacement field. The values
            should be given in voxel coordinates, and the grid is also assumed
            to be in voxel coordinates.
        interpolation_args: Arguments defining interpolation and extrapolation behavior
        data_format: Format of the provided data, either "displacement" or "coordinate"
    """

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "GridDeformation":
        if not isinstance(children["data"], MappableTensor):
            raise ValueError("Invalid children for samplable composable")
        return GridDeformation(
            data=children["data"],
            interpolation_args=self._interpolation_args,
        )

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        displacement = super().__call__(masked_coordinates)
        return displacement + masked_coordinates

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
        )

    def __repr__(self) -> str:
        return f"GridDeformation(data={self._data}, interpolation_args={self._interpolation_args})"


class _GridDeformationInverse(GridVolume):
    """Inverse of a GridDeformation

    The inverse is computed using a fixed point iteration
    """

    def __init__(
        self,
        inverted_data: MappableTensor,
        interpolation_args: InterpolationArgs,
        inversion_arguments: DeformationInversionArguments,
    ) -> None:
        super().__init__(
            data=inverted_data,
            interpolation_args=interpolation_args,
        )
        self._inversion_arguments = inversion_arguments

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_GridDeformationInverse":
        if not isinstance(children["data"], MappableTensor):
            raise ValueError("Invalid children for samplable composable")
        return _GridDeformationInverse(
            inverted_data=children["data"],
            interpolation_args=self._interpolation_args,
            inversion_arguments=self._inversion_arguments,
        )

    def __call__(
        self,
        masked_coordinates: MappableTensor,
    ) -> MappableTensor:
        coordinates_as_slice = masked_coordinates.as_slice(self._data.spatial_shape)
        data, data_mask = self._data.generate(generate_missing_mask=False, cast_mask=False)
        voxel_coordinates = masked_coordinates.generate_values()
        inverted_values = fixed_point_invert_deformation(
            displacement_field=data,
            arguments=self._inversion_arguments,
            initial_guess=(None if coordinates_as_slice is None else -data[coordinates_as_slice]),
            coordinates=voxel_coordinates,
        )
        mask = self._evaluate_mask(voxel_coordinates + inverted_values, data_mask)
        return PlainTensor(inverted_values, mask) + masked_coordinates

    def invert(self, **inversion_parameters) -> GridDeformation:
        return GridDeformation(
            data=self._data,
            interpolation_args=self._interpolation_args,
        )

    def __repr__(self) -> str:
        return (
            f"_GridCoordinateMappingInverse(displacement_field={self._data}, "
            f"interpolation_args={self._interpolation_args})"
        )


def create_volume(
    data: MappableTensor,
    coordinate_system: CoordinateSystem,
    interpolation_args: Optional[InterpolationArgs] = None,
) -> IComposableMapping:
    """Create volume based on grid samples"""
    grid_volume = GridVolume(
        data=data,
        interpolation_args=interpolation_args,
    )
    return grid_volume @ coordinate_system.to_voxel_coordinates


def create_deformation(
    data: MappableTensor,
    coordinate_system: CoordinateSystem,
    interpolation_args: Optional[InterpolationArgs] = None,
    *,
    data_format: str = "displacement",
    data_coordinates: str = "voxel",
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
        data_format: Format of the provided data, either "displacement" or "coordinate"
        data_coordinates: Format of the provided data coordinates, either "voxel" or "world"
    """
    if data_coordinates == "world":
        data = coordinate_system.to_voxel_coordinates(data)
    elif data_coordinates != "voxel":
        raise ValueError(f"Invalid data_coordinates: {data_coordinates}")
    if data_format == "coordinate":
        data = data - coordinate_system.voxel_grid()
    elif data_format != "displacement":
        raise ValueError(f"Invalid data_format: {data_format}")
    grid_volume = GridDeformation(
        data=data,
        interpolation_args=interpolation_args,
    )
    return (
        coordinate_system.from_voxel_coordinates
        @ grid_volume
        @ coordinate_system.to_voxel_coordinates
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
