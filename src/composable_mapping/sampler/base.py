"""Base sampler implementations"""

from abc import abstractmethod
from contextlib import AbstractContextManager
from math import ceil, floor
from threading import local
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

from torch import Tensor, ones
from torch.autograd.functional import vjp
from torch.nn.functional import conv1d, conv_transpose1d

from composable_mapping.mappable_tensor import MappableTensor, mappable
from composable_mapping.util import (
    crop_and_then_pad_spatial,
    get_spatial_dims,
    includes_padding,
    is_croppable_first,
)

from .convolution_sampling import (
    apply_flipping_permutation_to_volume,
    obtain_conv_parameters,
)
from .interface import DataFormat, ISampler, LimitDirection
from .inverse import FixedPointInverseSampler

if TYPE_CHECKING:
    from composable_mapping.coordinate_system import CoordinateSystem


_CONV_PARAMETERS_TYPE = Optional[
    Tuple[
        List[Optional[Tensor]],
        List[int],
        List[int],
        List[bool],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        List[int],
        List[int],
    ]
]


class ISeparableKernelSupport:
    """Interface for defining kernel supports for a kernel and its derivatives"""

    @abstractmethod
    def __call__(self, limit_direction: LimitDirection) -> Tuple[float, bool, bool]:
        """Interpolation kernel support for the convolution and whether min and max
        are inclusive or not"""

    @abstractmethod
    def derivative(self) -> "ISeparableKernelSupport":
        """Obtain kernel support function of the derivative kernel"""


class SymmetricPolynomialKernelSupport(ISeparableKernelSupport):
    """Kernel support function for polynomial kernels"""

    def __init__(
        self,
        kernel_width: float,
        polynomial_degree: int,
    ) -> None:
        self._kernel_width = kernel_width
        self._polynomial_degree = polynomial_degree

    def __call__(self, limit_direction: LimitDirection) -> Tuple[float, bool, bool]:
        if self._polynomial_degree < 0:
            # kernel is zeros
            return (1.0, True, False)
        bound_inclusive = self._polynomial_degree == 0
        if limit_direction == LimitDirection.left():
            return (self._kernel_width, False, bound_inclusive)
        if limit_direction == LimitDirection.right():
            return (self._kernel_width, bound_inclusive, False)
        if limit_direction == LimitDirection.average():
            return (self._kernel_width, bound_inclusive, bound_inclusive)
        raise ValueError("Unknown limit direction")

    @staticmethod
    def _update_polynomial_degree_func(
        spatial_dim: int, func: Callable[[int], int]
    ) -> Callable[[int], int]:
        def updated_func(spatial_dim_: int) -> int:
            if spatial_dim_ != spatial_dim:
                return func(spatial_dim_)
            return func(spatial_dim_) - 1

        return updated_func

    def derivative(self) -> "SymmetricPolynomialKernelSupport":
        return SymmetricPolynomialKernelSupport(
            kernel_width=self._kernel_width,
            polynomial_degree=self._polynomial_degree - 1,
        )


class BaseSeparableSampler(ISampler):
    """Base sampler in voxel coordinates which can be implemented as a
    separable convolution

    Arguments:
        extrapolation_mode: Extrapolation mode for out-of-bound coordinates.
        mask_extrapolated_regions_for_empty_volume_mask: Whether to mask
            extrapolated regions when input volume mask is empty.
        convolution_threshold: Maximum allowed difference in coordinates
            for using convolution-based sampling (the difference might be upper
            bounded when doing the decision).
        mask_threshold: Maximum allowed weight for masked regions in a
            sampled location to still consider it valid (non-masked).
        limit_direction: Direction for evaluating the kernel at
            discontinuous points.
    """

    def __init__(
        self,
        extrapolation_mode: str,
        mask_extrapolated_regions_for_empty_volume_mask: bool,
        convolution_threshold: float,
        mask_threshold: float,
        limit_direction: Union[LimitDirection, Callable[[int], LimitDirection]],
    ) -> None:
        if extrapolation_mode not in ("zeros", "border", "reflection"):
            raise ValueError("Unknown extrapolation mode")
        self._extrapolation_mode = extrapolation_mode
        self._mask_extrapolated_regions_for_empty_volume_mask = (
            mask_extrapolated_regions_for_empty_volume_mask
        )
        self._convolution_threshold = convolution_threshold
        self._mask_threshold = mask_threshold
        self._limit_direction = (
            limit_direction.for_all_spatial_dims()
            if isinstance(limit_direction, LimitDirection)
            else limit_direction
        )

    @abstractmethod
    def _kernel_support(self, spatial_dim: int) -> ISeparableKernelSupport:
        """Kernel support function for the interpolation kernel

        Args:
            spatial_dim: Spatial dimension for which to obtain the kernel support

        Returns:
            Kernel support function for the kernel.
        """

    @abstractmethod
    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
        """Is the kernel is interpolating.

        Args:
            spatial_dim: Spatial dimension for which to obtain the information.

        Returns:
            Whether the kernel is interpolating over the specified spatial dimension.
        """

    @abstractmethod
    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        """Evaluate the interpolation kernel for the given 1d coordinates with the
        discontinuous points evaluated as left limit.

        Args:
            coordinates: 1d coordinates with shape (n_coordinates,)
            spatial_dim: Spatial dimension for which to evaluate the kernel

        Returns:
            Interpolation kernel evaluated at the given coordinates.
        """

    @abstractmethod
    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        """Evaluate the interpolation kernel for the given 1d coordinates with the
        discontinuous points evaluated as right limit.

        Args:
            coordinates: 1d coordinates with shape (n_coordinates,)
            spatial_dim: Spatial dimension for which to evaluate the kernel

        Returns:
            Interpolation kernel evaluated at the given coordinates.
        """

    def _kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        if self._limit_direction(spatial_dim) == LimitDirection.left():
            return self._left_limit_kernel(coordinates, spatial_dim)
        if self._limit_direction(spatial_dim) == LimitDirection.right():
            return self._right_limit_kernel(coordinates, spatial_dim)
        if self._limit_direction(spatial_dim) == LimitDirection.average():
            return 0.5 * (
                self._left_limit_kernel(coordinates, spatial_dim)
                + self._right_limit_kernel(coordinates, spatial_dim)
            )
        raise ValueError("Unknown limit direction")

    def derivative(
        self,
        spatial_dim: int,
        limit_direction: Union[
            LimitDirection, Callable[[int], LimitDirection]
        ] = LimitDirection.average(),
    ) -> "GenericSeparableDerivativeSampler":
        return GenericSeparableDerivativeSampler(
            spatial_dim=spatial_dim,
            parent_left_limit_kernel=self._left_limit_kernel,
            parent_right_limit_kernel=self._right_limit_kernel,
            parent_kernel_support=self._kernel_support,
            parent_is_interpolating_kernel=self._is_interpolating_kernel,
            limit_direction=limit_direction,
            extrapolation_mode=self._extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                self._mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=self._convolution_threshold,
            mask_threshold=self._mask_threshold,
        )

    def __call__(self, volume: MappableTensor, coordinates: MappableTensor) -> MappableTensor:
        if coordinates.n_channel_dims != 1:
            raise ValueError("Interpolation assumes single channel coordinates")
        if coordinates.channels_shape[0] != len(volume.spatial_shape):
            raise ValueError("Interpolation assumes same number of channels as spatial dims")
        interpolated = self._interpolate_conv(volume, coordinates)
        if interpolated is None:
            return self._interpolate_general(volume, coordinates)
        return interpolated

    @property
    def _padding_mode_and_value(self) -> Tuple[str, float]:
        return {
            "zeros": ("constant", 0.0),
            "border": ("replicate", 0.0),
            "reflection": ("reflect", 0.0),
        }[self._extrapolation_mode]

    def _obtain_conv_parameters(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> _CONV_PARAMETERS_TYPE:
        if voxel_coordinates.displacements is not None:
            return None
        grid = voxel_coordinates.grid
        if grid is None:
            return None
        grid_affine_matrix = grid.affine_transformation.as_host_matrix()
        if grid_affine_matrix is None:
            return None
        conv_parameters = obtain_conv_parameters(
            volume_spatial_shape=volume.spatial_shape,
            grid_spatial_shape=grid.spatial_shape,
            grid_affine_matrix=grid_affine_matrix,
            is_interpolating_kernel=[
                self._is_interpolating_kernel(spatial_dim)
                for spatial_dim in range(len(grid.spatial_shape))
            ],
            kernel_support=[
                self._kernel_support(spatial_dim)(self._limit_direction(spatial_dim))
                for spatial_dim in range(len(grid.spatial_shape))
            ],
            convolution_threshold=self._convolution_threshold,
            target_device=voxel_coordinates.device,
        )
        if conv_parameters is None:
            return None
        (
            conv_kernel_coordinates,
            conv_strides,
            conv_paddings,
            transposed_convolve,
            pre_pads_or_crops,
            post_pads_or_crops,
            spatial_permutation,
            flipped_spatial_dims,
        ) = conv_parameters
        conv_kernels = [
            (
                self._kernel(
                    kernel_coordinates,
                    spatial_dim=spatial_permutation[spatial_dim],
                )
                if kernel_coordinates is not None
                else None
            )
            for spatial_dim, kernel_coordinates in enumerate(conv_kernel_coordinates)
        ]
        padding_mode, _padding_value = self._padding_mode_and_value
        if not is_croppable_first(
            spatial_shape=volume.spatial_shape, pads_or_crops=pre_pads_or_crops, mode=padding_mode
        ):
            return None
        return (
            conv_kernels,
            conv_strides,
            conv_paddings,
            transposed_convolve,
            pre_pads_or_crops,
            post_pads_or_crops,
            spatial_permutation,
            flipped_spatial_dims,
        )

    def _interpolate_conv(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> Optional[MappableTensor]:
        sampling_parameter_cache = _get_sampling_parameter_cache()
        if sampling_parameter_cache is not None:
            if sampling_parameter_cache.has_sampling_parameters():
                conv_parameters: _CONV_PARAMETERS_TYPE = (
                    sampling_parameter_cache.get_sampling_parameters()
                )
            else:
                conv_parameters = self._obtain_conv_parameters(
                    volume=volume, voxel_coordinates=voxel_coordinates
                )
                sampling_parameter_cache.append_sampling_parameters(conv_parameters)
        else:
            conv_parameters = self._obtain_conv_parameters(
                volume=volume, voxel_coordinates=voxel_coordinates
            )
        if conv_parameters is None:
            return None
        (
            conv_kernels,
            conv_strides,
            conv_paddings,
            transposed_convolve,
            pre_pads_or_crops,
            post_pads_or_crops,
            spatial_permutation,
            flipped_spatial_dims,
        ) = conv_parameters
        padding_mode, padding_value = self._padding_mode_and_value

        values, mask = volume.generate(
            generate_missing_mask=includes_padding(pre_pads_or_crops)
            and self._mask_extrapolated_regions_for_empty_volume_mask,
            cast_mask=False,
        )
        interpolated_values = apply_flipping_permutation_to_volume(
            values,
            n_channel_dims=volume.n_channel_dims,
            spatial_permutation=spatial_permutation,
            flipped_spatial_dims=flipped_spatial_dims,
        )
        interpolated_values = crop_and_then_pad_spatial(
            interpolated_values,
            pads_or_crops=pre_pads_or_crops,
            mode=padding_mode,
            value=padding_value,
            n_channel_dims=volume.n_channel_dims,
        )
        interpolated_values = self._separable_conv(
            interpolated_values,
            kernels=conv_kernels,
            strides=conv_strides,
            paddings=conv_paddings,
            transposed=transposed_convolve,
            n_channel_dims=volume.n_channel_dims,
        )
        interpolated_values = crop_and_then_pad_spatial(
            interpolated_values,
            pads_or_crops=post_pads_or_crops,
            mode=padding_mode,
            value=padding_value,
            n_channel_dims=volume.n_channel_dims,
        )
        if mask is None:
            interpolated_mask: Optional[Tensor] = None
        else:
            interpolated_mask = apply_flipping_permutation_to_volume(
                mask,
                n_channel_dims=volume.n_channel_dims,
                spatial_permutation=spatial_permutation,
                flipped_spatial_dims=flipped_spatial_dims,
            )
            interpolated_mask = crop_and_then_pad_spatial(
                interpolated_mask,
                pads_or_crops=pre_pads_or_crops,
                mode="constant",
                value=False,
                n_channel_dims=volume.n_channel_dims,
            )
            interpolated_mask = (
                self._separable_conv(
                    (~interpolated_mask).to(dtype=voxel_coordinates.dtype),
                    kernels=[None if kernel is None else kernel.abs() for kernel in conv_kernels],
                    strides=conv_strides,
                    paddings=conv_paddings,
                    transposed=transposed_convolve,
                    n_channel_dims=volume.n_channel_dims,
                )
                <= self._mask_threshold
            )
            interpolated_mask = crop_and_then_pad_spatial(
                interpolated_mask,
                pads_or_crops=post_pads_or_crops,
                mode="constant",
                value=False,
                n_channel_dims=volume.n_channel_dims,
            )
        return mappable(
            interpolated_values, interpolated_mask, n_channel_dims=volume.n_channel_dims
        )

    @staticmethod
    def _get_flooring_function(inclusive: bool) -> Callable[[Union[float, int]], int]:
        if inclusive:
            return lambda x: int(floor(x))
        return lambda x: int(ceil(x - 1))

    def _separable_conv(
        self,
        volume: Tensor,
        kernels: Sequence[Optional[Tensor]],
        strides: Sequence[int],
        paddings: Sequence[int],
        transposed: Sequence[bool],
        n_channel_dims: int,
    ) -> Tensor:
        if (
            len(kernels) != len(get_spatial_dims(volume.ndim, n_channel_dims))
            or len(kernels) != len(strides)
            or len(kernels) != len(transposed)
            or len(kernels) != len(paddings)
        ):
            raise ValueError("Invalid number of kernels, strides, transposed, or paddings")
        for spatial_dim, (kernel, stride, dim_transposed, padding) in enumerate(
            zip(kernels, strides, transposed, paddings)
        ):
            if kernel is None or kernel.size(0) == 1:
                assert dim_transposed is False
                dim = get_spatial_dims(volume.ndim, n_channel_dims)[spatial_dim]
                volume = volume[(slice(None),) * dim + (slice(None, None, stride),)]
                if kernel is not None:
                    volume = kernel * volume
            else:
                volume = self._conv1d(
                    volume,
                    kernel=kernel,
                    stride=stride,
                    padding=padding,
                    transposed=dim_transposed,
                    spatial_dim=spatial_dim,
                    n_channel_dims=n_channel_dims,
                )
        return volume

    @staticmethod
    def _conv1d(
        volume: Tensor,
        kernel: Tensor,
        stride: int,
        padding: int,
        transposed: bool,
        spatial_dim: int,
        n_channel_dims: int,
    ) -> Tensor:
        dim = get_spatial_dims(volume.ndim, n_channel_dims)[spatial_dim]
        volume = volume.moveaxis(dim, -1)
        dim_excluded_shape = volume.shape[:-1]
        volume = volume.reshape(-1, 1, volume.size(-1))
        conv_function = cast(Callable[..., Tensor], conv1d if not transposed else conv_transpose1d)
        convolved = conv_function(  # pylint: disable=not-callable
            volume,
            kernel[None, None],
            bias=None,
            stride=(stride,),
            padding=padding,
        ).reshape(dim_excluded_shape + (-1,))
        return convolved.moveaxis(-1, dim)

    def _interpolate_general(
        self, volume: MappableTensor, voxel_coordinates: MappableTensor
    ) -> MappableTensor:
        volume_values, volume_mask = volume.generate(
            generate_missing_mask=self._mask_extrapolated_regions_for_empty_volume_mask,
            cast_mask=False,
        )
        coordinate_values = voxel_coordinates.generate_values()
        interpolated_values = self.sample_values(volume_values, coordinate_values)
        if volume_mask is not None:
            interpolated_mask: Optional[Tensor] = self.sample_mask(
                volume_mask,
                coordinate_values,
            )
        else:
            interpolated_mask = None
        return mappable(
            interpolated_values, interpolated_mask, n_channel_dims=volume.n_channel_dims
        )

    def inverse(
        self,
        coordinate_system: "CoordinateSystem",
        data_format: DataFormat,
        arguments: Optional[Mapping[str, Any]] = None,
    ) -> ISampler:
        if data_format.coordinate_type == "voxel" and data_format.representation == "displacements":
            if arguments is None:
                arguments = {}
            fixed_point_inversion_arguments = arguments.get("fixed_point_inversion_arguments", {})
            return FixedPointInverseSampler(
                self,
                forward_solver=fixed_point_inversion_arguments.get("forward_solver"),
                backward_solver=fixed_point_inversion_arguments.get("backward_solver"),
                forward_dtype=fixed_point_inversion_arguments.get("forward_dtype"),
                backward_dtype=fixed_point_inversion_arguments.get("backward_dtype"),
                mask_extrapolated_regions_for_empty_volume_mask=(
                    self._mask_extrapolated_regions_for_empty_volume_mask
                ),
            )
        raise ValueError(
            "Inverse sampler has been currently implemented only for voxel "
            "displacements data format."
        )


class GenericSeparableDerivativeSampler(BaseSeparableSampler):
    """Sampler for sampling spatial derivatives of a separable kernel
    sampler.

    Args:
        spatial_dim: Spatial dimension over which the derivative is taken.
        parent_left_limit_kernel: Parent sampler's left limit kernel function.
        parent_right_limit_kernel: Parent sampler's right limit kernel function.
        parent_kernel_support: Parent sampler's kernel support function.
        parent_is_interpolating_kernel: Parent sampler's information on whether the kernel
            is interpolating.
        limit_direction: Direction for evaluating the kernel at discontinuous points.
        extrapolation_mode: Extrapolation mode for out-of-bound coordinates.
        mask_extrapolated_regions_for_empty_volume_mask: Whether to mask
            extrapolated regions when input volume mask is empty.
        convolution_threshold: Maximum allowed difference in coordinates
            for using convolution-based sampling.
        mask_threshold: Maximum allowed weight for masked regions in a
            sampled location to still consider it valid (non-masked).

    """

    def __init__(
        self,
        spatial_dim: int,
        parent_left_limit_kernel: Callable[[Tensor, int], Tensor],
        parent_right_limit_kernel: Callable[[Tensor, int], Tensor],
        parent_kernel_support: Callable[[int], ISeparableKernelSupport],
        parent_is_interpolating_kernel: Callable[[int], bool],
        limit_direction: Union[LimitDirection, Callable[[int], LimitDirection]],
        extrapolation_mode: str = "border",
        mask_extrapolated_regions_for_empty_volume_mask: bool = True,
        convolution_threshold: float = 1e-4,
        mask_threshold: float = 1e-5,
    ) -> None:
        super().__init__(
            extrapolation_mode=extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=convolution_threshold,
            mask_threshold=mask_threshold,
            limit_direction=limit_direction,
        )
        self._spatial_dim = spatial_dim
        self._parent_left_limit_kernel = parent_left_limit_kernel
        self._parent_right_limit_kernel = parent_right_limit_kernel
        self._parent_is_interpolating_kernel = parent_is_interpolating_kernel
        self._parent_kernel_support = parent_kernel_support

    def _kernel_support(self, spatial_dim: int) -> ISeparableKernelSupport:
        if spatial_dim == self._spatial_dim:
            return self._parent_kernel_support(spatial_dim).derivative()
        return self._parent_kernel_support(spatial_dim)

    def derivative(
        self,
        spatial_dim: int,
        limit_direction: Union[
            LimitDirection, Callable[[int], LimitDirection]
        ] = LimitDirection.average(),
    ) -> "GenericSeparableDerivativeSampler":
        return GenericSeparableDerivativeSampler(
            spatial_dim=spatial_dim,
            parent_left_limit_kernel=self._left_limit_kernel,
            parent_right_limit_kernel=self._right_limit_kernel,
            parent_kernel_support=self._kernel_support,
            parent_is_interpolating_kernel=self._is_interpolating_kernel,
            limit_direction=limit_direction,
            extrapolation_mode=self._extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                self._mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=self._convolution_threshold,
            mask_threshold=self._mask_threshold,
        )

    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
        if spatial_dim == self._spatial_dim:
            return False
        return self._parent_is_interpolating_kernel(spatial_dim)

    def _derive_function(
        self,
        kernel_function: Callable[[Tensor, int], Tensor],
        coordinates: Tensor,
        spatial_dim: int,
    ) -> Tensor:
        def partial_kernel_function(coordinates: Tensor) -> Tensor:
            return -kernel_function(coordinates, spatial_dim)

        _output, derivatives = vjp(
            partial_kernel_function,
            inputs=coordinates,
            v=ones(coordinates.shape, device=coordinates.device, dtype=coordinates.dtype),
            create_graph=coordinates.requires_grad,
        )
        return derivatives

    def _kernel_derivative(
        self,
        kernel_function: Callable[[Tensor, int], Tensor],
        coordinates: Tensor,
        spatial_dim: int,
    ) -> Tensor:
        if spatial_dim == self._spatial_dim:
            return self._derive_function(kernel_function, coordinates, spatial_dim)
        return kernel_function(coordinates, spatial_dim)

    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        return self._kernel_derivative(
            self._parent_left_limit_kernel, coordinates=coordinates, spatial_dim=spatial_dim
        )

    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        return self._kernel_derivative(
            self._parent_right_limit_kernel, coordinates=coordinates, spatial_dim=spatial_dim
        )

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            "Sampling derivatives at arbitrary coordinates is not currently implemented. "
            "Contact the developers if you would like this to be included. Currently only "
            "cases where the coordinates are on a regular grid with the axes "
            "aligned with the spatial dimensions are supported (implementable with convolution)."
        )

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            "Sampling derivatives at arbitrary coordinates is not currently implemented. "
            "Contact the developers if you would like this to be included. Currently only "
            "cases where the coordinates are on a regular grid with the axes "
            "aligned with the spatial dimensions are supported (implementable with convolution)."
        )


_SAMPLING_PARAMETER_CACHE_STACK = local()
_SAMPLING_PARAMETER_CACHE_STACK.stack = []


def _get_sampling_parameter_cache() -> Optional["EnumeratedSamplingParameterCache"]:
    if _SAMPLING_PARAMETER_CACHE_STACK.stack:
        return _SAMPLING_PARAMETER_CACHE_STACK.stack[-1]
    return None


class EnumeratedSamplingParameterCache(AbstractContextManager):
    """Context manager for caching convolution parameters for sampling
    with separable kernels.

    Is intended for situations where same order of sampling operations is
    repeated multiple times, e.g. when iterating a training step.

    One can then use this context manager to cache the convolution parameters
    for the sampling operations and avoid recomputing them. That can save
    few milliseconds per sampling operation.
    """

    def __init__(self) -> None:
        self._sampling_parameters: List[Any] = []
        self._index: Optional[int] = None

    def __enter__(self) -> None:
        self._index = 0
        _SAMPLING_PARAMETER_CACHE_STACK.stack.append(self)

    def has_sampling_parameters(self) -> bool:
        """Check if there are any sampling parameters in the cache"""
        if self._index is None:
            raise ValueError("Cache not active.")
        return self._index < len(self._sampling_parameters)

    def get_sampling_parameters(self) -> Optional[Any]:
        """Get the next set of sampling parameters from the cache"""
        if self._index is None:
            raise ValueError("Cache not active.")
        sampling_parameters = self._sampling_parameters[self._index]
        self._index += 1
        return sampling_parameters

    def append_sampling_parameters(self, sampling_parameters: Any) -> None:
        """Append the next set of sampling parameters to the cache"""
        self._sampling_parameters.append(sampling_parameters)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        _SAMPLING_PARAMETER_CACHE_STACK.stack.pop()
        self._index = None
