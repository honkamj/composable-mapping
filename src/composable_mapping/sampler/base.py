"""Base sampler implementations"""

from abc import abstractmethod
from math import ceil, floor
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

from numpy import argsort
from torch import Tensor, contiguous_format
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import empty, linspace, long, ones, tensor, zeros
from torch.autograd.functional import vjp
from torch.nn.functional import conv1d, conv_transpose1d

from composable_mapping.mappable_tensor import (
    DiagonalAffineMatrixDefinition,
    HostAffineTransformation,
    HostDiagonalAffineTransformation,
    IdentityAffineTransformation,
    IHostAffineTransformation,
    MappableTensor,
    mappable,
)
from composable_mapping.util import (
    crop_and_then_pad_spatial,
    get_spatial_dims,
    includes_padding,
    is_croppable_first,
)

from .interface import DataFormat, ISampler, LimitDirection
from .inverse import FixedPointInverseSampler

if TYPE_CHECKING:
    from composable_mapping.coordinate_system import CoordinateSystem


class IKernelSupport:
    """Interface for defining kernel supports for a kernel and its derivatives"""

    @abstractmethod
    def __call__(
        self, spatial_dim: int, limit_direction: LimitDirection
    ) -> Tuple[float, bool, bool]:
        """Return the kernel size for the convolution and whether min and max
        are inclusive or not"""

    @abstractmethod
    def derivative(self, spatial_dim: int) -> "IKernelSupport":
        """Return the kernel support function of the derivative kernel"""


class SymmetricPolynomialKernelSupport(IKernelSupport):
    """Kernel support function for polynomial kernels"""

    def __init__(
        self,
        kernel_width: Callable[[int], float],
        polynomial_degree: Callable[[int], int],
    ) -> None:
        self._kernel_width = kernel_width
        self._polynomial_degree = polynomial_degree

    def __call__(
        self, spatial_dim: int, limit_direction: LimitDirection
    ) -> Tuple[float, bool, bool]:
        degree = self._polynomial_degree(spatial_dim)
        if degree < 0:
            return (1.0, False, False)
        bound_inclusive = degree == 0
        if limit_direction == LimitDirection.LEFT:
            return (self._kernel_width(spatial_dim), False, bound_inclusive)
        if limit_direction == LimitDirection.RIGHT:
            return (self._kernel_width(spatial_dim), bound_inclusive, False)
        if limit_direction == LimitDirection.AVERAGE:
            return (self._kernel_width(spatial_dim), bound_inclusive, bound_inclusive)
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

    def derivative(self, spatial_dim: int) -> "SymmetricPolynomialKernelSupport":
        return SymmetricPolynomialKernelSupport(
            kernel_width=self._kernel_width,
            polynomial_degree=self._update_polynomial_degree_func(
                spatial_dim, self._polynomial_degree
            ),
        )


class BaseSeparableSampler(ISampler):
    """Base sampler in voxel coordinates which can be implemented as a
    separable convolution"""

    def __init__(
        self,
        extrapolation_mode: str,
        mask_extrapolated_regions_for_empty_volume_mask: bool,
        convolution_threshold: float,
        mask_threshold: float,
        interpolating_sampler: bool,
        kernel_support: IKernelSupport,
        limit_direction: LimitDirection,
    ) -> None:
        if extrapolation_mode not in ("zeros", "border", "reflection"):
            raise ValueError("Unknown extrapolation mode")
        self._extrapolation_mode = extrapolation_mode
        self._mask_extrapolated_regions_for_empty_volume_mask = (
            mask_extrapolated_regions_for_empty_volume_mask
        )
        self._convolution_threshold = convolution_threshold
        self._mask_threshold = mask_threshold
        self._interpolating_sampler = interpolating_sampler
        self._kernel_support = kernel_support
        self._limit_direction = limit_direction

    @abstractmethod
    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        """Return the interpolation kernel for the given 1d coordinates with the
        discontinuous points evaluated as the left limit"""

    @abstractmethod
    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        """Return the interpolation kernel for the given 1d coordinates with the
        discontinuous points evaluated as the right limit"""

    def _kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        if self._limit_direction == LimitDirection.LEFT:
            return self._left_limit_kernel(coordinates, spatial_dim)
        if self._limit_direction == LimitDirection.RIGHT:
            return self._right_limit_kernel(coordinates, spatial_dim)
        if self._limit_direction == LimitDirection.AVERAGE:
            return 0.5 * (
                self._left_limit_kernel(coordinates, spatial_dim)
                + self._right_limit_kernel(coordinates, spatial_dim)
            )
        raise ValueError("Unknown limit direction")

    def derivative(
        self, spatial_dim: int, limit_direction: LimitDirection = LimitDirection.AVERAGE
    ) -> "ISampler":
        return GenericSeparableDerivativeSampler(
            spatial_dim=spatial_dim,
            left_limit_kernel=lambda coordinates, spatial_dim, _: self._left_limit_kernel(
                coordinates, spatial_dim
            ),
            right_limit_kernel=lambda coordinates, spatial_dim, _: self._right_limit_kernel(
                coordinates, spatial_dim
            ),
            kernel_support=self._kernel_support.derivative(spatial_dim),
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

    @staticmethod
    def _obtain_flipping_permutation(
        matrix: Tensor,
    ) -> Optional["_FlippingPermutation"]:
        largest_indices = matrix[:-1, :-1].abs().argmax(dim=1).tolist()
        n_dims = matrix.shape[-1] - 1
        if not _FlippingPermutation.is_valid_permutation(n_dims, largest_indices):
            return None
        permutation = tuple(argsort(largest_indices))
        flipped_spatial_dims = [
            largest_index
            for column, largest_index in enumerate(largest_indices)
            if matrix[largest_index, column] < 0
        ]
        return _FlippingPermutation(permutation, flipped_spatial_dims)

    def _extract_conv_interpolatable_parameters(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, "_FlippingPermutation"]]:
        if voxel_coordinates.displacements is not None:
            return None
        grid = voxel_coordinates.grid
        assert grid is not None
        affine_transformation = grid.affine_transformation
        channels_shape = affine_transformation.channels_shape
        if channels_shape[0] != channels_shape[1]:
            return None
        n_dims = len(grid.spatial_shape)
        host_matrix = grid.affine_transformation.as_host_matrix()
        if host_matrix is None:
            return None
        host_matrix = host_matrix.view(-1, n_dims + 1, n_dims + 1)
        flipping_permutation = self._obtain_flipping_permutation(host_matrix[0])
        if flipping_permutation is None:
            return None
        affine_transformation = (
            flipping_permutation.as_transformation(spatial_shape=volume.spatial_shape)
            @ affine_transformation
        )
        host_matrix = affine_transformation.as_host_matrix()
        assert host_matrix is not None
        host_matrix = host_matrix.view(-1, n_dims + 1, n_dims + 1)
        diagonal = host_matrix[0, :-1, :-1].diagonal()
        if diagonal.any() == 0.0:
            return None
        translation = host_matrix[0, :-1, -1].clone(memory_format=contiguous_format)

        transposed_convolve = diagonal < 0.75
        downsampling_factor = empty(diagonal.shape, device=diagonal.device, dtype=diagonal.dtype)
        downsampling_factor[~transposed_convolve] = diagonal[~transposed_convolve].round()
        downsampling_factor[transposed_convolve] = 1 / (1 / diagonal[transposed_convolve]).round()
        rounded_diagonal_matrix = DiagonalAffineMatrixDefinition(
            diagonal=downsampling_factor, translation=translation
        ).as_matrix()
        difference_matrix = host_matrix[:, :-1, :-1] - rounded_diagonal_matrix[:-1, :-1]
        shape_tensor = tensor(grid.spatial_shape, device=torch_device("cpu"), dtype=grid.dtype)
        max_conv_coordinate_difference_upper_bound = (
            (difference_matrix * shape_tensor).abs().amax(dim=(0, 2))
        )
        if (max_conv_coordinate_difference_upper_bound > self._convolution_threshold).any():
            return None
        if self._interpolating_sampler and not transposed_convolve.all():
            rounded_translation = translation.round()
            max_slicing_coordinate_difference_upper_bound = (
                (translation - rounded_translation).abs()
                + max_conv_coordinate_difference_upper_bound
            ).amax()
            convolve = (
                max_slicing_coordinate_difference_upper_bound > self._convolution_threshold
            ) & (~transposed_convolve)
            no_convolve_or_transposed_convolve = (~transposed_convolve) & (~convolve)
            translation[no_convolve_or_transposed_convolve] = translation[
                no_convolve_or_transposed_convolve
            ].round()
        else:
            convolve = ~transposed_convolve
        return downsampling_factor, translation, convolve, transposed_convolve, flipping_permutation

    def _obtain_conv_parameters(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> Optional[
        Tuple[
            Sequence[Optional[Tensor]],
            Sequence[int],
            Sequence[int],
            Sequence[bool],
            Sequence[Tuple[int, int]],
            Sequence[Tuple[int, int]],
            "_FlippingPermutation",
        ]
    ]:
        conv_interpolation_parameters = self._extract_conv_interpolatable_parameters(
            volume, voxel_coordinates
        )
        if conv_interpolation_parameters is None:
            return None
        downsampling_factor, translation, convolve, transposed_convolve, flipping_permutation = (
            conv_interpolation_parameters
        )
        pre_pads_or_crops: List[Tuple[int, int]] = []
        post_pads_or_crops: List[Tuple[int, int]] = []
        conv_paddings: List[int] = []
        conv_kernels: List[Optional[Tensor]] = []
        for spatial_dim, (
            dim_size_volume,
            dim_size_grid,
            dim_convolve,
            dim_transposed_convolve,
            dim_translation,
            dim_downsampling_factor,
        ) in enumerate(
            zip(
                flipping_permutation.permute_sequence(volume.spatial_shape),
                voxel_coordinates.spatial_shape,
                convolve.tolist(),
                transposed_convolve.tolist(),
                translation.tolist(),
                downsampling_factor.tolist(),
            )
        ):
            (kernel_width, inclusive_min, inclusive_max) = self._kernel_support(
                spatial_dim, self._limit_direction
            )
            lower_flooring_function = self._get_flooring_function(inclusive_min)
            upper_flooring_function = self._get_flooring_function(inclusive_max)
            min_coordinate = dim_translation
            max_coordinate = dim_translation + dim_downsampling_factor * (dim_size_grid - 1)
            if dim_convolve or dim_transposed_convolve:
                pre_pad_or_crop_lower = lower_flooring_function(kernel_width / 2 - dim_translation)
                pre_pad_or_crop_upper = upper_flooring_function(
                    kernel_width / 2
                    + dim_translation
                    + dim_downsampling_factor * (dim_size_grid - 1)
                    - (dim_size_volume - 1)
                )
                if dim_transposed_convolve:
                    start_kernel_coordinate = (
                        1
                        - dim_translation
                        + upper_flooring_function(
                            (kernel_width / 2 - (1 - dim_translation)) / dim_downsampling_factor
                        )
                        * dim_downsampling_factor
                    )
                    end_kernel_coordinate = (
                        -dim_translation
                        - lower_flooring_function(
                            (kernel_width / 2 - dim_translation) / dim_downsampling_factor
                        )
                        * dim_downsampling_factor
                    )
                    kernel_step_size = dim_downsampling_factor
                else:
                    relative_coordinate = dim_translation - floor(dim_translation)
                    start_kernel_coordinate = (
                        -lower_flooring_function(kernel_width / 2 - relative_coordinate)
                        - relative_coordinate
                    )
                    end_kernel_coordinate = upper_flooring_function(
                        kernel_width / 2 - (1 - relative_coordinate)
                    ) + (1 - relative_coordinate)
                    kernel_step_size = 1
                kernel_coordinates = linspace(
                    start_kernel_coordinate,
                    end_kernel_coordinate,
                    int(
                        round(
                            abs(end_kernel_coordinate - start_kernel_coordinate) / kernel_step_size
                        )
                    )
                    + 1,
                    dtype=voxel_coordinates.dtype,
                    device=voxel_coordinates.device,
                )
                kernel = self._kernel(kernel_coordinates, spatial_dim=spatial_dim)
            else:
                kernel = None
                pre_pad_or_crop_lower = -int(min_coordinate)
                pre_pad_or_crop_upper = int(max_coordinate) - (dim_size_volume - 1)
            if dim_transposed_convolve:
                post_pad_or_crop_lower = -int(
                    round(
                        (min_coordinate + pre_pad_or_crop_lower + start_kernel_coordinate)
                        / dim_downsampling_factor
                    )
                )
                post_pad_or_crop_upper = -int(
                    round(
                        (
                            (dim_size_volume - 1)
                            + pre_pad_or_crop_upper
                            - end_kernel_coordinate
                            - max_coordinate
                        )
                        / dim_downsampling_factor
                    )
                )
                assert post_pad_or_crop_lower <= 0 and post_pad_or_crop_upper <= 0
                conv_padding = -max(post_pad_or_crop_lower, post_pad_or_crop_upper)
                post_pad_or_crop_lower += conv_padding
                post_pad_or_crop_upper += conv_padding
            else:
                conv_padding = 0
                post_pad_or_crop_lower = 0
                post_pad_or_crop_upper = 0
            conv_kernels.append(kernel)
            conv_paddings.append(conv_padding)
            pre_pads_or_crops.append((pre_pad_or_crop_lower, pre_pad_or_crop_upper))
            post_pads_or_crops.append((post_pad_or_crop_lower, post_pad_or_crop_upper))
        padding_mode, _padding_value = self._padding_mode_and_value
        if not is_croppable_first(
            spatial_shape=volume.spatial_shape, pads_or_crops=pre_pads_or_crops, mode=padding_mode
        ):
            return None
        conv_strides = (
            (
                downsampling_factor * (~transposed_convolve)
                + (1 / downsampling_factor) * transposed_convolve
            )
            .round()
            .to(dtype=long)
        )
        return (
            conv_kernels,
            conv_strides.tolist(),
            conv_paddings,
            transposed_convolve.tolist(),
            pre_pads_or_crops,
            post_pads_or_crops,
            flipping_permutation,
        )

    def _interpolate_conv(
        self,
        volume: MappableTensor,
        voxel_coordinates: MappableTensor,
    ) -> Optional[MappableTensor]:
        conv_parameters = self._obtain_conv_parameters(volume, voxel_coordinates)
        if conv_parameters is None:
            return None
        (
            conv_kernels,
            conv_strides,
            conv_paddings,
            transposed_convolve,
            pre_pads_or_crops,
            post_pads_or_crops,
            flipping_permutation,
        ) = conv_parameters
        padding_mode, padding_value = self._padding_mode_and_value

        values, mask = volume.generate(
            generate_missing_mask=includes_padding(pre_pads_or_crops)
            and self._mask_extrapolated_regions_for_empty_volume_mask,
            cast_mask=False,
        )
        interpolated_values = flipping_permutation(values, n_channel_dims=volume.n_channel_dims)
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
            interpolated_mask = flipping_permutation(mask, n_channel_dims=volume.n_channel_dims)
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

    @staticmethod
    def _get_flooring_function(inclusive: bool) -> Callable[[Union[float, int]], int]:
        if inclusive:
            return lambda x: int(floor(x))
        return lambda x: int(ceil(x - 1))

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
            return FixedPointInverseSampler(
                self,
                forward_solver=arguments.get("forward_solver"),
                backward_solver=arguments.get("backward_solver"),
                forward_dtype=arguments.get("forward_dtype"),
                backward_dtype=arguments.get("backward_dtype"),
                mask_extrapolated_regions_for_empty_volume_mask=(
                    self._mask_extrapolated_regions_for_empty_volume_mask
                ),
            )
        raise ValueError(
            "Inverse sampler has been currently implemented only for voxel "
            "displacements data format."
        )


class GenericSeparableDerivativeSampler(BaseSeparableSampler):
    """Base implementation of a separable derivative sampler"""

    def __init__(
        self,
        spatial_dim: int,
        left_limit_kernel: Callable[[Tensor, int, bool], Tensor],
        right_limit_kernel: Callable[[Tensor, int, bool], Tensor],
        kernel_support: IKernelSupport,
        limit_direction: LimitDirection,
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
            interpolating_sampler=False,
            kernel_support=kernel_support,
            limit_direction=limit_direction,
        )
        self._spatial_dim = spatial_dim
        self._parent_left_limit_kernel = left_limit_kernel
        self._parent_right_limit_kernel = right_limit_kernel

    def derivative(
        self, spatial_dim: int, limit_direction: LimitDirection = LimitDirection.AVERAGE
    ) -> "ISampler":
        return GenericSeparableDerivativeSampler(
            spatial_dim=spatial_dim,
            left_limit_kernel=self._left_limit_kernel_for_derivation,
            right_limit_kernel=self._right_limit_kernel_for_derivation,
            kernel_support=self._kernel_support.derivative(spatial_dim),
            limit_direction=limit_direction,
            extrapolation_mode=self._extrapolation_mode,
            mask_extrapolated_regions_for_empty_volume_mask=(
                self._mask_extrapolated_regions_for_empty_volume_mask
            ),
            convolution_threshold=self._convolution_threshold,
            mask_threshold=self._mask_threshold,
        )

    def _derive_function(
        self,
        kernel_function: Callable[[Tensor, int, bool], Tensor],
        coordinates: Tensor,
        spatial_dim: int,
        needs_derivatives: bool,
    ) -> Tensor:
        def partial_kernel_function(coordinates: Tensor) -> Tensor:
            return kernel_function(coordinates, spatial_dim, True)

        _output, derivatives = vjp(
            partial_kernel_function,
            inputs=coordinates,
            v=ones(coordinates.shape),
            create_graph=needs_derivatives,
        )
        return derivatives

    def _kernel_for_derivation(
        self,
        kernel_function: Callable[[Tensor, int, bool], Tensor],
        coordinates: Tensor,
        spatial_dim: int,
        needs_derivatives: bool,
    ) -> Tensor:
        if spatial_dim == self._spatial_dim:
            return self._derive_function(
                kernel_function, coordinates, spatial_dim, needs_derivatives=needs_derivatives
            )
        return kernel_function(coordinates, spatial_dim, needs_derivatives)

    def _left_limit_kernel_for_derivation(
        self, coordinates: Tensor, spatial_dim: int, needs_derivatives: bool
    ) -> Tensor:
        return self._kernel_for_derivation(
            self._parent_left_limit_kernel, coordinates, spatial_dim, needs_derivatives
        )

    def _right_limit_kernel_for_derivation(
        self, coordinates: Tensor, spatial_dim: int, needs_derivatives: bool
    ) -> Tensor:
        return self._kernel_for_derivation(
            self._parent_right_limit_kernel, coordinates, spatial_dim, needs_derivatives
        )

    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        return self._left_limit_kernel_for_derivation(
            coordinates, spatial_dim, needs_derivatives=False
        )

    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        return self._right_limit_kernel_for_derivation(
            coordinates, spatial_dim, needs_derivatives=False
        )

    def sample_values(
        self,
        volume: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            "Sampling values at arbitrary coordinates is not currently implemented "
            "for derivatives."
        )

    def sample_mask(
        self,
        mask: Tensor,
        coordinates: Tensor,
    ) -> Tensor:
        raise NotImplementedError(
            "Sampling values at arbitrary coordinates is not currently implemented "
            "for derivatives."
        )


class _FlippingPermutation:
    def __init__(
        self, spatial_permutation: Sequence[int], flipped_spatial_dims: Sequence[int]
    ) -> None:
        self.spatial_permutation = spatial_permutation
        self.flipped_spatial_dims = flipped_spatial_dims

    @staticmethod
    def is_valid_permutation(n_dims: int, spatial_permutation: Sequence[int]) -> bool:
        """Check if the permutation is valid"""
        return set(spatial_permutation) == set(range(n_dims))

    def permute_sequence(self, sequence: Sequence) -> Tuple:
        """Permute a sequence"""
        if len(sequence) != len(self.spatial_permutation):
            raise ValueError("Sequence has wrong length")
        return tuple(sequence[spatial_dim] for spatial_dim in self.spatial_permutation)

    def __call__(self, volume: Tensor, n_channel_dims: int) -> Tensor:
        spatial_dims = get_spatial_dims(volume.ndim, n_channel_dims=n_channel_dims)
        if self.flipped_spatial_dims:
            flipped_dims = [spatial_dims[spatial_dim] for spatial_dim in self.flipped_spatial_dims]
            volume = volume.flip(dims=flipped_dims)
        volume = volume.permute(tuple(range(spatial_dims[0])) + self.permute_sequence(spatial_dims))
        return volume

    def _flipping_transformation(
        self,
        spatial_shape: Sequence[int],
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> HostDiagonalAffineTransformation:
        n_dims = len(spatial_shape)
        diagonal = ones(n_dims, dtype=dtype, device=torch_device("cpu"))
        translation = zeros(n_dims, dtype=dtype, device=torch_device("cpu"))
        for flipped_spatial_dim in self.flipped_spatial_dims:
            diagonal[flipped_spatial_dim] = -1.0
            translation[flipped_spatial_dim] = spatial_shape[flipped_spatial_dim] - 1
        return HostDiagonalAffineTransformation(
            diagonal=diagonal,
            translation=translation,
            device=device,
        )

    def as_transformation(
        self,
        spatial_shape: Sequence[int],
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> IHostAffineTransformation:
        """Return the transformation corresponding to the flipping and permutation"""
        if self.spatial_permutation == tuple(range(len(spatial_shape))):
            if not self.flipped_spatial_dims:
                return IdentityAffineTransformation(
                    n_dims=len(spatial_shape), dtype=dtype, device=device
                )
            return self._flipping_transformation(spatial_shape, dtype=dtype, device=device)
        matrix = self._flipping_transformation(
            spatial_shape, dtype=dtype, device=device
        ).as_matrix()
        matrix = matrix[tuple(self.spatial_permutation) + (-1,), :]
        return HostAffineTransformation(transformation_matrix_on_host=matrix, device=device)
