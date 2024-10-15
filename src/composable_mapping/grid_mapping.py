"""Grid based mappings"""

from contextlib import ContextDecorator
from functools import partial
from threading import local
from typing import Mapping, Optional

from deformation_inversion_layer import (
    DeformationInversionArguments,
    fixed_point_invert_deformation,
)
from torch import Tensor

from .base import BaseComposableMapping
from .interface import IComposableMapping, IInterpolator
from .interpolator import LinearInterpolator
from .mappable_tensor import MappableTensor, PlainTensor
from .tensor_like import ITensorLike


class VoxelGridVolume(BaseComposableMapping):
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
        interpolator: Optional[IInterpolator] = None,
    ) -> None:
        self._data = data
        self._interpolator = get_interpolator(interpolator)

    @property
    def data(self) -> MappableTensor:
        """Get the data"""
        return self._data

    @property
    def interpolator(self) -> IInterpolator:
        """Get the interpolator"""
        return self._interpolator

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {
            "data": self._data,
        }

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "VoxelGridVolume":
        if not isinstance(children["data"], MappableTensor):
            raise ValueError("Invalid children for VoxelGridVolume")
        return VoxelGridVolume(
            data=children["data"],
            interpolator=self._interpolator,
        )

    def __call__(self, coordinates: MappableTensor) -> MappableTensor:
        return self._interpolator(self._data, coordinates)

    def invert(self, **inversion_parameters) -> IComposableMapping:
        raise NotImplementedError("No inversion implemented for grid volumes")

    def __repr__(self) -> str:
        return f"VoxelGridVolume(data={self._data}, interpolator={self._interpolator}, "


class VoxelGridDeformation(VoxelGridVolume):
    """Continuously defined deformation based on regular grid of displacements

    Arguments:
        data: (batch_size, n_dims, dim_1, ..., dim_{n_dims}). The values are
            given as displacements in voxel coordinates.
        interpolation_args: Arguments defining interpolation and extrapolation behavior
        data_format: Format of the provided data, either "displacement" or "coordinate"
    """

    def __init__(
        self,
        displacements: MappableTensor,
        interpolator: Optional[IInterpolator] = None,
    ) -> None:
        if displacements.n_channel_dims != 1:
            raise ValueError("Displacement field must have a single channel")
        if displacements.channels_shape[0] != len(displacements.spatial_shape):
            raise ValueError("Displacement field must have same number of channels as spatial dims")
        super().__init__(
            data=displacements,
            interpolator=interpolator,
        )

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "VoxelGridDeformation":
        if not isinstance(children["data"], MappableTensor):
            raise ValueError("Invalid children for VoxelGridDeformation")
        return VoxelGridDeformation(
            displacements=children["data"],
            interpolator=self._interpolator,
        )

    def __call__(self, masked_coordinates: MappableTensor) -> MappableTensor:
        displacement = super().__call__(masked_coordinates)
        return displacement + masked_coordinates

    def invert(self, **inversion_parameters) -> "_VoxelGridDeformationInverse":
        """Invert the deformation

        inversion_parameters:
            fixed_point_inversion_arguments (Mapping[str, Any]): Arguments for
                fixed point inversion
        """
        fixed_point_inversion_arguments = inversion_parameters.get(
            "fixed_point_inversion_arguments", {}
        )
        default_interpolator = partial(self._interpolator.interpolate_values)
        deformation_inversion_arguments = DeformationInversionArguments(
            interpolator=fixed_point_inversion_arguments.get("interpolator", default_interpolator),
            forward_solver=fixed_point_inversion_arguments.get("forward_solver"),
            backward_solver=fixed_point_inversion_arguments.get("backward_solver"),
            forward_dtype=fixed_point_inversion_arguments.get("forward_dtype"),
            backward_dtype=fixed_point_inversion_arguments.get("backward_dtype"),
        )
        return _VoxelGridDeformationInverse(
            displacements_to_invert=self._data,
            interpolator=self._interpolator,
            inversion_arguments=deformation_inversion_arguments,
        )

    def __repr__(self) -> str:
        return f"GridDeformation(data={self._data}, interpolator={self._interpolator})"


class _VoxelGridDeformationInverse(VoxelGridVolume):
    """Inverse of a GridDeformation

    The inverse is computed using a fixed point iteration
    """

    def __init__(
        self,
        displacements_to_invert: MappableTensor,
        interpolator: IInterpolator,
        inversion_arguments: DeformationInversionArguments,
    ) -> None:
        super().__init__(
            data=displacements_to_invert,
            interpolator=interpolator,
        )
        self._inversion_arguments = inversion_arguments

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_VoxelGridDeformationInverse":
        if not isinstance(children["data"], MappableTensor):
            raise ValueError("Invalid children for samplable composable")
        return _VoxelGridDeformationInverse(
            displacements_to_invert=children["data"],
            interpolator=self._interpolator,
            inversion_arguments=self._inversion_arguments,
        )

    def __call__(
        self,
        masked_coordinates: MappableTensor,
    ) -> MappableTensor:
        coordinates_as_slice = masked_coordinates.as_slice(self._data.spatial_shape)
        data, data_mask = self._data.generate(generate_missing_mask=True, cast_mask=False)
        voxel_coordinates = masked_coordinates.generate_values()
        inverted_values = fixed_point_invert_deformation(
            displacement_field=data,
            arguments=self._inversion_arguments,
            initial_guess=(None if coordinates_as_slice is None else -data[coordinates_as_slice]),
            coordinates=voxel_coordinates,
        )
        mask = self._interpolator.interpolate_mask(
            mask=data_mask,
            voxel_coordinates=voxel_coordinates,
        )
        return PlainTensor(inverted_values, mask) + masked_coordinates

    def invert(self, **inversion_parameters) -> VoxelGridDeformation:
        return VoxelGridDeformation(
            displacements=self._data,
            interpolator=self._interpolator,
        )

    def __repr__(self) -> str:
        return (
            f"_VoxelGridDeformationInverse(displacement_field_to_invert={self._data}, "
            f"interpolator={self._interpolator}, inversion_arguments={self._inversion_arguments})"
        )


_DEFAULT_INTERPOLATOR: Optional[IInterpolator] = None
_DEFAULT_INTERPOLATOR_CONTEXT_STACK = local()
_DEFAULT_INTERPOLATOR_CONTEXT_STACK.stack = []


def get_default_interpolator() -> IInterpolator:
    """Get current default interpolation args"""
    if _DEFAULT_INTERPOLATOR_CONTEXT_STACK.stack:
        return _DEFAULT_INTERPOLATOR_CONTEXT_STACK.stack[-1]
    if _DEFAULT_INTERPOLATOR is None:
        return LinearInterpolator()
    return _DEFAULT_INTERPOLATOR


def set_default_interpolator(interpolator: Optional[IInterpolator]) -> None:
    """Set default interpolation args"""
    global _DEFAULT_INTERPOLATOR  # pylint: disable=global-statement
    _DEFAULT_INTERPOLATOR = interpolator


def clear_default_interpolation_args() -> None:
    """Clear default interpolation args"""
    global _DEFAULT_INTERPOLATOR  # pylint: disable=global-statement
    _DEFAULT_INTERPOLATOR = None


def get_interpolator(interpolator: Optional[IInterpolator]) -> IInterpolator:
    """Get interpolation args, either from argument or default"""
    return interpolator if interpolator is not None else get_default_interpolator()


class default_interpolation_args(  # this is supposed to appear as function - pylint: disable=invalid-name
    ContextDecorator
):
    """Context manager for setting default interpolation args"""

    def __init__(self, interpolator: IInterpolator):
        self.interpolator = interpolator

    def __enter__(self) -> None:
        _DEFAULT_INTERPOLATOR_CONTEXT_STACK.stack.append(self.interpolator)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _DEFAULT_INTERPOLATOR_CONTEXT_STACK.stack.pop()
