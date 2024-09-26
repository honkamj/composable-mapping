"""Affine transformation implementations"""

from typing import List, Literal, Mapping, Optional, Sequence

from torch import Tensor, allclose, cat
from torch import device as torch_device
from torch import diag_embed
from torch import dtype as torch_dtype
from torch import eye, get_default_dtype, inverse, matmul, ones, zeros
from torch.jit import script

from .base import BaseComposableMapping, BaseTensorLikeWrapper
from .interface import (
    IAffineTransformation,
    IComposableMapping,
    IMaskedTensor,
    ITensorLike,
)
from .util import (
    broadcast_tensors_around_channel_dims,
    broadcast_to_shape_around_channel_dims,
    channels_last,
    get_channels_shape,
    index_by_channel_dims,
    merge_batch_dimensions,
    move_channels_first,
    move_channels_last,
    unmerge_batch_dimensions,
)


class BaseAffineTransformation(IAffineTransformation):
    """Base affine transformation"""

    def __matmul__(self, affine_transformation: IAffineTransformation) -> "IAffineTransformation":
        if isinstance(affine_transformation, IdentityAffineTransformation):
            if self.n_input_dims != affine_transformation.n_output_dims:
                raise ValueError("Transformation dimensionalities do not match")
            return self
        if isinstance(self, IdentityAffineTransformation):
            if self.n_input_dims != affine_transformation.n_output_dims:
                raise ValueError("Transformation dimensionalities do not match")
            return affine_transformation
        return AffineTransformation(
            compose_affine_transformation_matrices(
                self.as_matrix(),
                affine_transformation.as_matrix(),
            )
        )

    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        return AffineTransformation(self.as_matrix() + affine_transformation.as_matrix())

    def __sub__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        return AffineTransformation(self.as_matrix() - affine_transformation.as_matrix())

    def __neg__(self) -> IAffineTransformation:
        return AffineTransformation(-self.as_matrix())


class AffineTransformation(BaseAffineTransformation, BaseTensorLikeWrapper):
    """Represents generic affine transformation

    Arguments:
        transformation_matrix: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, ...),
            if None, corresponds to identity transformation
    """

    def __init__(self, transformation_matrix: Tensor) -> None:
        self._transformation_matrix = transformation_matrix

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {"transformation_matrix": self._transformation_matrix}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "AffineTransformation":
        return AffineTransformation(tensors["transformation_matrix"])

    def is_zero(self, n_input_dims: int, n_output_dims: int) -> Optional[bool]:
        if self.n_input_dims != n_input_dims or self.n_output_dims != n_output_dims:
            raise ValueError("Transformation dimensionalities do not match")
        if self._transformation_matrix.device == torch_device("cpu"):
            return is_zero_matrix(self._transformation_matrix)
        return None

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        return _broadcast_and_transform_values(
            self._transformation_matrix,
            values,
            n_channel_dims=n_channel_dims,
        )

    def invert(self) -> "AffineTransformation":
        """Invert the transformation"""
        return AffineTransformation(channels_last(2, 2)(inverse)(self._transformation_matrix))

    def as_matrix(
        self,
    ) -> Tensor:
        return self._transformation_matrix

    def as_cpu_matrix(self) -> Optional[Tensor]:
        if self._transformation_matrix.device == torch_device("cpu"):
            return self._transformation_matrix
        return None

    @property
    def n_input_dims(self) -> int:
        return (
            self._transformation_matrix.size(
                index_by_channel_dims(
                    n_total_dims=self._transformation_matrix.ndim,
                    channel_dim_index=1,
                    n_channel_dims=2,
                )
            )
            - 1
        )

    @property
    def n_output_dims(self) -> int:
        return (
            self._transformation_matrix.size(
                index_by_channel_dims(
                    n_total_dims=self._transformation_matrix.ndim,
                    channel_dim_index=0,
                    n_channel_dims=2,
                )
            )
            - 1
        )

    def __repr__(self) -> str:
        return f"AffineTransformation(transformation_matrix={self._transformation_matrix})"


class IdentityAffineTransformation(BaseAffineTransformation):
    """Identity transformation"""

    def __init__(
        self,
        n_dims: int,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> None:
        self._n_dims = n_dims
        self._dtype = get_default_dtype() if dtype is None else dtype
        self._device = torch_device("cpu") if device is None else device

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        if (
            get_coordinates_affine_dimensionality(get_channels_shape(values.shape, n_channel_dims))
            != self.n_input_dims
        ):
            raise ValueError(
                "Coordinates have wrong dimensionality for the identity transformation"
            )
        return values

    def is_zero(self, n_input_dims: int, n_output_dims: int) -> Literal[False]:
        if self._n_dims != n_input_dims or self._n_dims != n_output_dims:
            raise ValueError("Transformation dimensionalities do not match")
        return False

    def invert(self) -> "IdentityAffineTransformation":
        """Invert the transformation"""
        return IdentityAffineTransformation(
            n_dims=self._n_dims, dtype=self._dtype, device=self._device
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return eye(self._n_dims + 1, device=self._device, dtype=self._dtype)

    def as_cpu_matrix(self) -> Tensor:
        return eye(self._n_dims + 1, device=torch_device("cpu"), dtype=self._dtype)

    @property
    def dtype(
        self,
    ) -> torch_dtype:
        return self._dtype

    @property
    def device(
        self,
    ) -> torch_device:
        return self._device

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "IdentityAffineTransformation":
        return IdentityAffineTransformation(
            n_dims=self._n_dims,
            dtype=self._dtype if dtype is None else dtype,
            device=self._device if device is None else device,
        )

    def detach(self) -> "IdentityAffineTransformation":
        return self

    @property
    def n_input_dims(self) -> int:
        return self._n_dims

    @property
    def n_output_dims(self) -> int:
        return self._n_dims

    def __repr__(self) -> str:
        return (
            f"IdentityAffineTransformation("
            f"n_dims={self._n_dims}, "
            f"dtype={self._dtype}, "
            f"device={self._device})"
        )


class ZeroAffineTransformation(BaseAffineTransformation):
    """Zero transformation"""

    def __init__(
        self,
        n_input_dims: int,
        n_output_dims: Optional[int] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> None:
        self._n_input_dims = n_input_dims
        self._n_output_dims = n_input_dims if n_output_dims is None else n_output_dims
        self._dtype = get_default_dtype() if dtype is None else dtype
        self._device = torch_device("cpu") if device is None else device

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        if (
            get_coordinates_affine_dimensionality(get_channels_shape(values.shape, n_channel_dims))
            != self.n_input_dims
        ):
            raise ValueError("Coordinates have wrong dimensionality for the zero transformation")
        if n_channel_dims == 1:
            shape_modification_dim = index_by_channel_dims(
                n_total_dims=len(values.shape), channel_dim_index=0, n_channel_dims=1
            )
        else:
            shape_modification_dim = index_by_channel_dims(
                n_total_dims=len(values.shape),
                channel_dim_index=-2,
                n_channel_dims=n_channel_dims,
            )
        target_shape = list(values.shape)
        target_shape[shape_modification_dim] = self.n_output_dims
        return zeros(target_shape, dtype=self._dtype, device=self._device)

    def is_zero(self, n_input_dims: int, n_output_dims: int) -> Literal[True]:
        if self._n_input_dims != n_input_dims or self._n_output_dims != n_output_dims:
            raise ValueError("Transformation dimensionalities do not match")
        return True

    def invert(self) -> "IdentityAffineTransformation":
        """Invert the transformation"""
        raise RuntimeError("Zero transformation is not invertible")

    def as_matrix(
        self,
    ) -> Tensor:
        matrix = zeros(
            self._n_input_dims + 1, self._n_output_dims + 1, device=self._device, dtype=self._dtype
        )
        matrix[-1, -1] = 1
        return matrix

    def as_cpu_matrix(self) -> Tensor:
        matrix = zeros(
            self._n_input_dims + 1,
            self._n_output_dims + 1,
            device=torch_device("cpu"),
            dtype=self._dtype,
        )
        matrix[-1, -1] = 1
        return matrix

    @property
    def dtype(
        self,
    ) -> torch_dtype:
        return self._dtype

    @property
    def device(
        self,
    ) -> torch_device:
        return self._device

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "ZeroAffineTransformation":
        return ZeroAffineTransformation(
            n_input_dims=self._n_input_dims,
            n_output_dims=self._n_output_dims,
            dtype=self._dtype if dtype is None else dtype,
            device=self._device if device is None else device,
        )

    def detach(self) -> "ZeroAffineTransformation":
        return self

    @property
    def n_input_dims(self) -> int:
        return self._n_input_dims

    @property
    def n_output_dims(self) -> int:
        return self._n_output_dims

    def __repr__(self) -> str:
        return (
            f"ZeroAffineTransformation("
            f"n_input_dims={self._n_input_dims}, "
            f"n_output_dims={self._n_output_dims}, "
            f"dtype={self._dtype}, "
            f"device={self._device})"
        )


class CPUAffineTransformation(BaseAffineTransformation):
    """Affine tranformation for which compositions and inversions are done
    actively on CPU, and the same computations on target devices are done only
    if needed

    Allows to do control flow decisions on CPU based on the actively computed
    transformation matrix without having to do CPU-GPU synchronization.

    Arguments:
        transformation_matrix_on_cpu: Transformation matrix on cpu
        device: Device to use for the transformation matrix produced by as_matrix method
    """

    def __init__(
        self,
        transformation_matrix_on_cpu: Tensor,
        device: Optional[torch_device] = None,
    ) -> None:
        if transformation_matrix_on_cpu.device != torch_device("cpu"):
            raise ValueError("Please give the matrix on CPU")
        if transformation_matrix_on_cpu.requires_grad:
            raise ValueError("The implementation assumes a detached transformation matrix.")
        self._transformation_matrix_cpu = transformation_matrix_on_cpu
        self._device = torch_device("cpu") if device is None else device
        if self._device != torch_device("cpu"):
            self._transformation_matrix_target_device = (
                self._transformation_matrix_cpu.pin_memory().to(
                    device=self._device, non_blocking=True
                )
            )
        else:
            self._transformation_matrix_target_device = transformation_matrix_on_cpu

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        if (
            get_coordinates_affine_dimensionality(get_channels_shape(values.shape, n_channel_dims))
            != self.n_input_dims
        ):
            raise ValueError("Coordinates have wrong dimensionality")
        if is_identity_matrix(self._transformation_matrix_cpu):
            return values
        return _broadcast_and_transform_values(self.as_matrix(), values)

    def is_zero(self, n_input_dims: int, n_output_dims: int) -> bool:
        if self.n_input_dims != n_input_dims or self.n_output_dims != n_output_dims:
            raise ValueError("Transformation dimensionalities do not match")
        return is_zero_matrix(self._transformation_matrix_cpu)

    def as_matrix(
        self,
    ) -> Tensor:
        return self._transformation_matrix_cpu.to(
            device=self._device, non_blocking=self._device != torch_device("cpu")
        )

    def as_cpu_matrix(self) -> Tensor:
        return self._transformation_matrix_cpu

    def invert(self) -> "CPUAffineTransformation":
        return _CPUAffineTransformationInverse(
            inverted_transformation_matrix_cpu=channels_last(2, 2)(inverse)(
                self._transformation_matrix_cpu
            ),
            transformation_to_invert=self,
        )

    def detach(self) -> "CPUAffineTransformation":
        return self

    def __matmul__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if isinstance(affine_transformation, CPUAffineTransformation):
            return _CPUAffineTransformationComposition(
                compsed_transformation_matrix_cpu=compose_affine_transformation_matrices(
                    self._transformation_matrix_cpu, affine_transformation.as_cpu_matrix()
                ),
                left_transformation=self,
                right_transformation=affine_transformation,
            )
        return super().__matmul__(affine_transformation)

    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if isinstance(affine_transformation, CPUAffineTransformation):
            return _CPUAffineTransformationSum(
                summed_transformation_matrix_cpu=self._transformation_matrix_cpu
                + affine_transformation.as_cpu_matrix(),
                left_transformation=self,
                right_transformation=affine_transformation,
            )
        return super().__add__(affine_transformation)

    def __sub__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if isinstance(affine_transformation, CPUAffineTransformation):
            negated_affine_transformation = -affine_transformation
            return _CPUAffineTransformationSum(
                summed_transformation_matrix_cpu=self._transformation_matrix_cpu
                + negated_affine_transformation.as_cpu_matrix(),
                left_transformation=self,
                right_transformation=negated_affine_transformation,
            )
        return super().__sub__(affine_transformation)

    def __neg__(self) -> "_CPUAffineTransformationNegation":
        return _CPUAffineTransformationNegation(
            negated_transformation_matrix_cpu=-self._transformation_matrix_cpu,
            negated_transformation=self,
        )

    def reduce(self) -> "CPUAffineTransformation":
        """Reduce the transformation to non-lazy version"""
        return CPUAffineTransformation(
            self.as_cpu_matrix(),
            device=self._device,
        )

    @property
    def n_input_dims(self) -> int:
        return (
            self._transformation_matrix_cpu.size(
                index_by_channel_dims(
                    n_total_dims=self._transformation_matrix_cpu.ndim,
                    channel_dim_index=1,
                    n_channel_dims=2,
                )
            )
            - 1
        )

    @property
    def n_output_dims(self) -> int:
        return (
            self._transformation_matrix_cpu.size(
                index_by_channel_dims(
                    n_total_dims=self._transformation_matrix_cpu.ndim,
                    channel_dim_index=0,
                    n_channel_dims=2,
                )
            )
            - 1
        )

    @property
    def dtype(
        self,
    ) -> torch_dtype:
        return self._transformation_matrix_cpu.dtype

    @property
    def device(
        self,
    ) -> torch_device:
        return self._device

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "CPUAffineTransformation":
        return CPUAffineTransformation(
            transformation_matrix_on_cpu=self._transformation_matrix_cpu.to(
                dtype=self._transformation_matrix_cpu.dtype if dtype is None else dtype,
            ),
            device=self._device if device is None else device,
        )

    def __repr__(self) -> str:
        return (
            f"CPUAffineTransformation("
            f"transformation_matrix_cpu={self._transformation_matrix_cpu}, "
            f"device={self._device}, "
            f"pin_memory={self._transformation_matrix_cpu.is_pinned()})"
        )


class _CPUAffineTransformationInverse(CPUAffineTransformation):
    def __init__(
        self,
        inverted_transformation_matrix_cpu: Tensor,
        transformation_to_invert: CPUAffineTransformation,
    ) -> None:
        super().__init__(
            inverted_transformation_matrix_cpu,
            device=transformation_to_invert.device,
        )
        self._transformation_to_invert = transformation_to_invert

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "_CPUAffineTransformationInverse":
        return _CPUAffineTransformationInverse(
            inverted_transformation_matrix_cpu=self._transformation_matrix_cpu.to(
                dtype=self._transformation_matrix_cpu.dtype if dtype is None else dtype,
            ),
            transformation_to_invert=self._transformation_to_invert.cast(
                dtype=dtype, device=device
            ),
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return channels_last(2, 2)(inverse)(self._transformation_to_invert.as_matrix())

    def __repr__(self) -> str:
        return (
            f"_CPUAffineTransformationInverse("
            f"inverted_transformation_matrix_cpu={self._transformation_matrix_cpu}, "
            f"transformation_to_invert={self._transformation_to_invert})"
        )


class _CPUAffineTransformationComposition(CPUAffineTransformation):
    def __init__(
        self,
        compsed_transformation_matrix_cpu: Tensor,
        left_transformation: CPUAffineTransformation,
        right_transformation: CPUAffineTransformation,
    ) -> None:
        super().__init__(
            compsed_transformation_matrix_cpu,
            device=left_transformation.device,
        )
        self._left_transformation = left_transformation
        self._right_transformation = right_transformation

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "_CPUAffineTransformationComposition":
        return _CPUAffineTransformationComposition(
            compsed_transformation_matrix_cpu=self._transformation_matrix_cpu.to(
                dtype=self._transformation_matrix_cpu.dtype if dtype is None else dtype,
            ),
            left_transformation=self._left_transformation.cast(dtype=dtype, device=device),
            right_transformation=self._right_transformation.cast(dtype=dtype, device=device),
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return compose_affine_transformation_matrices(
            self._left_transformation.as_matrix(), self._right_transformation.as_matrix()
        )

    def __repr__(self) -> str:
        return (
            f"_CPUAffineTransformationComposition("
            f"compsed_transformation_matrix_cpu={self._transformation_matrix_cpu}, "
            f"left_transformation={self._left_transformation}, "
            f"right_transformation={self._right_transformation})"
        )


class _CPUAffineTransformationSum(CPUAffineTransformation):
    def __init__(
        self,
        summed_transformation_matrix_cpu: Tensor,
        left_transformation: CPUAffineTransformation,
        right_transformation: CPUAffineTransformation,
    ) -> None:
        super().__init__(
            summed_transformation_matrix_cpu,
            device=left_transformation.device,
        )
        self._left_transformation = left_transformation
        self._right_transformation = right_transformation

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "_CPUAffineTransformationSum":
        return _CPUAffineTransformationSum(
            summed_transformation_matrix_cpu=self._transformation_matrix_cpu.to(
                dtype=self._transformation_matrix_cpu.dtype if dtype is None else dtype,
            ),
            left_transformation=self._left_transformation.cast(dtype=dtype, device=device),
            right_transformation=self._right_transformation.cast(dtype=dtype, device=device),
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return self._left_transformation.as_matrix() + self._right_transformation.as_matrix()

    def __repr__(self) -> str:
        return (
            f"_CPUAffineTransformationSum("
            f"summed_transformation_matrix_cpu={self._transformation_matrix_cpu}, "
            f"left_transformation={self._left_transformation}, "
            f"right_transformation={self._right_transformation})"
        )


class _CPUAffineTransformationNegation(CPUAffineTransformation):
    def __init__(
        self,
        negated_transformation_matrix_cpu: Tensor,
        negated_transformation: CPUAffineTransformation,
    ) -> None:
        super().__init__(
            negated_transformation_matrix_cpu,
            device=negated_transformation.device,
        )
        self._negated_transformation = negated_transformation

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "_CPUAffineTransformationNegation":
        return _CPUAffineTransformationNegation(
            negated_transformation_matrix_cpu=self._transformation_matrix_cpu.to(
                dtype=self._transformation_matrix_cpu.dtype if dtype is None else dtype,
            ),
            negated_transformation=self._negated_transformation.cast(dtype=dtype, device=device),
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return -self._negated_transformation.as_matrix()

    def __repr__(self) -> str:
        return (
            f"_CPUAffineTransformationNegation("
            f"negated_transformation_matrix_cpu={self._transformation_matrix_cpu}, "
            f"negated_transformation={self._negated_transformation}"
        )


class ComposableAffine(BaseComposableMapping):
    """Composable wrapper for affine transformations"""

    def __init__(self, affine_transformation: IAffineTransformation) -> None:
        self._affine_transformation = affine_transformation

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return masked_coordinates.apply_affine(self._affine_transformation)

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"affine_transformation": self._affine_transformation}

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "ComposableAffine":
        if not isinstance(children["affine_transformation"], IAffineTransformation):
            raise ValueError("Child of a composable affine must be an affine transformation")
        return ComposableAffine(children["affine_transformation"])

    def invert(self, **inversion_parameters) -> IComposableMapping:
        return ComposableAffine(self._affine_transformation.invert())

    def __repr__(self) -> str:
        return f"ComposableAffine(affine_transformation={self._affine_transformation})"


class ComposableVoxelGridAffine(BaseComposableMapping):
    """Composable wrapper for affine transformations of voxel grid of masked tensors"""

    def __init__(self, affine_transformation: IAffineTransformation) -> None:
        self._affine_transformation = affine_transformation

    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        return masked_coordinates.add_grid(self._affine_transformation)

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"affine_transformation": self._affine_transformation}

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "ComposableVoxelGridAffine":
        if not isinstance(children["affine_transformation"], IAffineTransformation):
            raise ValueError("Child of a composable affine must be an affine transformation")
        return ComposableVoxelGridAffine(children["affine_transformation"])

    def invert(self, **inversion_parameters) -> IComposableMapping:
        return ComposableVoxelGridAffine(-self._affine_transformation)

    def __repr__(self) -> str:
        return f"ComposableAffine(affine_transformation={self._affine_transformation})"


class NotAffineTransformationError(Exception):
    """Error raised when a composable mapping is not affine"""


def as_affine_transformation(
    composable_mapping: IComposableMapping, n_dims: int
) -> IAffineTransformation:
    """Extract affine mapping from composable mapping

    Raises an error if the composable mapping is not fully affine.
    """
    tracer = _AffineTracer(
        IdentityAffineTransformation(
            n_dims, dtype=composable_mapping.dtype, device=composable_mapping.device
        )
    )
    traced = composable_mapping(tracer)
    if isinstance(traced, _AffineTracer):
        return traced.affine_transformation
    raise NotAffineTransformationError("Could not infer affine transformation")


class _AffineTracer(IMaskedTensor, BaseTensorLikeWrapper):
    """Can be used to trace affine component of a composable mapping"""

    def __init__(self, affine_transformation: IAffineTransformation) -> None:
        self.affine_transformation = affine_transformation

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"affine_transformation": self.affine_transformation}

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "_AffineTracer":
        if not isinstance(children["affine_transformation"], IAffineTransformation):
            raise ValueError("Child of a composable affine must be an affine transformation")
        return _AffineTracer(children["affine_transformation"])

    def generate(
        self,
        generate_missing_mask: bool = True,
        cast_mask: bool = False,
    ):
        raise NotAffineTransformationError(
            "Affine tracer has no values or mask! Usually this error means that "
            "the traced mapping is not affine."
        )

    def generate_mask(
        self,
        generate_missing_mask: bool = True,
        cast_mask: bool = False,
    ):
        raise NotAffineTransformationError(
            "Affine tracer has no mask! Usually this error means that "
            "the traced mapping is not affine."
        )

    def generate_values(
        self,
    ) -> Tensor:
        raise NotAffineTransformationError(
            "Affine tracer has no values! Usually this error means that "
            "the traced mapping is not affine."
        )

    def add_grid(self, affine_transformation: IAffineTransformation) -> IMaskedTensor:
        raise NotAffineTransformationError(
            "Affine tracer has no voxel grid! Usually this error means that "
            "the traced mapping is not affine."
        )

    def displace(self, displacement: Tensor) -> IMaskedTensor:
        raise NotAffineTransformationError(
            "Affine tracer can not be displaced! Usually this error means that "
            "the traced mapping is not affine."
        )

    def apply_affine(self, affine_transformation: IAffineTransformation) -> "IMaskedTensor":
        return _AffineTracer(affine_transformation @ self.affine_transformation)

    def has_mask(self) -> bool:
        return False

    @property
    def channels_shape(self) -> Sequence[int]:
        raise NotAffineTransformationError(
            "Affine tracer has no channels! Usually this error means that "
            "the traced mapping is not affine."
        )

    @property
    def shape(self) -> Sequence[int]:
        raise NotAffineTransformationError(
            "Affine tracer has no shape! Usually this error means that "
            "the traced mapping is not affine."
        )

    @property
    def spatial_shape(self) -> Sequence[int]:
        raise NotAffineTransformationError(
            "Affine tracer has no spatial shape! Usually this error means that "
            "the traced mapping is not affine."
        )

    def clear_mask(self) -> "_AffineTracer":
        return self

    def detach(self) -> "_AffineTracer":
        return self

    def as_slice(self, target_shape: Sequence[int]) -> None:
        return None

    def reduce(self) -> IMaskedTensor:
        return _AffineTracer(IdentityAffineTransformation(self.affine_transformation.n_input_dims))

    def __repr__(self) -> str:
        return f"_AffineTracer(affine_transformation={self.affine_transformation})"

    def modify(self, values: Tensor, mask: Optional[Tensor]) -> IMaskedTensor:
        raise NotAffineTransformationError(
            "Affine tracer has no values or mask! Usually this error means that "
            "the traced mapping is not affine."
        )

    def modify_values(self, values: Tensor) -> IMaskedTensor:
        raise NotAffineTransformationError(
            "Affine tracer has no values! Usually this error means that "
            "the traced mapping is not affine."
        )

    def modify_mask(self, mask: Optional[Tensor]) -> IMaskedTensor:
        raise NotAffineTransformationError(
            "Affine tracer has no mask! Usually this error means that "
            "the traced mapping is not affine."
        )


def compose_affine_transformation_matrices(
    *transformations: Tensor,
) -> Tensor:
    """Compose two transformation matrices

    Args:
        transformations: Tensors with shape ([batch_size, ]n_dims + 1, n_dims + 1, *)

    Returns: transformation_1: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, *)
    """
    if len(transformations) == 0:
        raise ValueError("At least one transformation matrix must be given.")
    composition = move_channels_last(transformations[0], 2)
    for transformation in transformations[1:]:
        composition = matmul(composition, move_channels_last(transformation, 2))
    return move_channels_first(composition, 2)


@script
def embed_transformation(matrix: Tensor, target_shape: List[int]) -> Tensor:
    """Embed transformation into larger dimensional space

    Args:
        matrix: Tensor with shape ([batch_size, ]n_dims, n_dims, ...)
        target_shape: Target matrix shape

    Returns: Tensor with shape (batch_size, *target_shape)
    """
    if len(target_shape) != 2:
        raise ValueError("Matrix shape must be two dimensional.")
    matrix = move_channels_last(matrix, 2)
    matrix, batch_dimensions_shape = merge_batch_dimensions(matrix, 2)
    batch_size = matrix.size(0)
    n_rows_needed = target_shape[0] - matrix.size(1)
    n_cols_needed = target_shape[1] - matrix.size(2)
    if n_rows_needed == 0 and n_cols_needed == 0:
        return matrix
    rows = cat(
        [
            zeros(
                n_rows_needed,
                min(matrix.size(2), matrix.size(1)),
                device=matrix.device,
                dtype=matrix.dtype,
            ),
            eye(
                n_rows_needed,
                max(0, matrix.size(2) - matrix.size(1)),
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ],
        dim=1,
    ).expand(batch_size, -1, -1)
    cols = cat(
        [
            zeros(
                min(target_shape[0], matrix.size(2)),
                n_cols_needed,
                device=matrix.device,
                dtype=matrix.dtype,
            ),
            eye(
                max(0, target_shape[0] - matrix.size(2)),
                n_cols_needed,
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ],
        dim=0,
    ).expand(batch_size, -1, -1)
    embedded_matrix = cat([cat([matrix, rows], dim=1), cols], dim=2)
    embedded_matrix = unmerge_batch_dimensions(
        embedded_matrix, batch_dimensions_shape=batch_dimensions_shape, n_channel_dims=2
    )
    return move_channels_first(embedded_matrix, 2)


def _convert_to_homogenous_coordinates(coordinates: Tensor, dim: int = -1) -> Tensor:
    """Converts the coordinates to homogenous coordinates

    Args:
        coordinates: Tensor with shape
            (dim_1, ..., dim_{dim}, ..., dim_{n_dims})

    Returns: Tensor with shape
        (dim_1, ..., dim_{dim + 1}, ..., dim_{n_dims})
    """
    if dim < 0:
        dim = coordinates.ndim + dim
    homogenous_coordinates = cat(
        [
            coordinates,
            ones(1, device=coordinates.device, dtype=coordinates.dtype).expand(
                *coordinates.shape[:dim], 1, *coordinates.shape[dim + 1 :]
            ),
        ],
        dim=dim,
    )
    return homogenous_coordinates


def generate_translation_matrix(translations: Tensor) -> Tensor:
    """Generator homogenous translation matrix with given translations

    Args:
        translations: Tensor with shape (batch_size, n_dims, ...)

    Returns: Tensor with shape (batch_size, n_dims + 1, n_dims + 1, ...)
    """
    translations = move_channels_last(translations, n_channel_dims=1)
    translations, batch_dimensions_shape = merge_batch_dimensions(translations, n_channel_dims=1)
    batch_size = translations.size(0)
    n_dims = translations.size(1)
    homogenous_translation = _convert_to_homogenous_coordinates(translations, dim=-1)
    translation_matrix = cat(
        [
            cat(
                [
                    eye(n_dims, device=translations.device, dtype=translations.dtype),
                    zeros(1, n_dims, device=translations.device, dtype=translations.dtype),
                ],
                dim=0,
            ).expand(batch_size, -1, -1),
            homogenous_translation[..., None],
        ],
        dim=2,
    ).view(-1, n_dims + 1, n_dims + 1)
    translation_matrix = unmerge_batch_dimensions(
        translation_matrix, batch_dimensions_shape=batch_dimensions_shape, n_channel_dims=2
    )
    return move_channels_first(translation_matrix, 2)


@script
def generate_scale_matrix(
    scales: Tensor,
) -> Tensor:
    """Generator scale matrix from given scales

    Args:
        scales: Tensor with shape (batch_size, n_scale_and_shear_axes, ...)

    Returns: Tensor with shape (batch_size, n_dims, n_dims, ...)
    """
    scales = move_channels_last(scales, n_channel_dims=1)
    scale_matrix = diag_embed(scales)
    return move_channels_first(scale_matrix, n_channel_dims=2)


def is_zero_matrix(matrix: Tensor) -> bool:
    """Return whether a matrix or batch of matrices is a zero matrix"""
    return allclose(
        matrix,
        zeros(
            1,
            dtype=matrix.dtype,
            device=matrix.device,
        ),
    )


def is_identity_matrix(matrix: Tensor) -> bool:
    """Return whether a matrix or batch of matrices is an identity"""
    if matrix.size(-2) != matrix.size(-1):
        return False
    first_channel_dim = index_by_channel_dims(
        n_total_dims=matrix.ndim, channel_dim_index=0, n_channel_dims=2
    )
    n_rows = matrix.size(first_channel_dim)
    identity_matrix = eye(
        n_rows,
        dtype=matrix.dtype,
        device=matrix.device,
    )
    broadcasted_identity_matrix = broadcast_to_shape_around_channel_dims(
        identity_matrix, shape=matrix.shape, n_channel_dims=2
    )
    return allclose(
        matrix,
        broadcasted_identity_matrix,
    )


def get_coordinates_affine_dimensionality(channels_shape: Sequence[int]) -> int:
    """Get the dimensionality of the affine transformation compatible with the coordinates"""
    if len(channels_shape) == 1:
        return channels_shape[0]
    return channels_shape[-2]


def _broadcast_and_transform_values(
    transformation_matrix: Tensor, values: Tensor, n_channel_dims: int = 1
) -> Tensor:
    values, transformation_matrix = broadcast_tensors_around_channel_dims(
        (values, transformation_matrix), n_channel_dims=(n_channel_dims, 2)
    )
    transformation_matrix = move_channels_last(transformation_matrix, 2)
    if n_channel_dims > 2:
        transformation_matrix = transformation_matrix[
            (...,) + (None,) * (n_channel_dims - 2) + 2 * (slice(None),)
        ]
    values = move_channels_last(values, n_channel_dims)
    if n_channel_dims == 1:
        values = values[..., None]
    transformed = matmul(
        transformation_matrix,
        _convert_to_homogenous_coordinates(values, dim=-2),
    )[..., :-1, :]
    if n_channel_dims == 1:
        transformed = transformed[..., 0]
    transformed = move_channels_first(transformed, n_channel_dims)

    return transformed
