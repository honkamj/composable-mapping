"""Affine transformation implementations"""

from typing import Mapping, Optional, Sequence

from torch import Tensor, allclose, cat
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import eye, get_default_dtype, inverse, matmul, ones

from composable_mapping.interface import ITensorLike

from .base import BaseComposableMapping, BaseTensorLikeWrapper
from .interface import IAffineTransformation, IComposableMapping, IMaskedTensor
from .util import (
    broadcast_tensors_around_channel_dims,
    broadcast_to_shape_around_channel_dims,
    channels_last,
    index_by_channel_dims,
    merge_batch_dimensions,
    merged_batch_dimensions,
    move_channels_first,
    move_channels_last,
    unmerge_batch_dimensions,
)


class BaseAffineTransformation(IAffineTransformation):
    """Base affine transformation"""

    def compose(self, affine_transformation: IAffineTransformation) -> "IAffineTransformation":
        if isinstance(affine_transformation, IdentityAffineTransformation):
            return self
        return AffineTransformation(
            compose_affine_transformation_matrices(
                self.as_matrix(),
                affine_transformation.as_matrix(),
            )
        )


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

    def __call__(self, coordinates: Tensor) -> Tensor:
        return _broadcast_and_transform_coordinates(coordinates, self._transformation_matrix)

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
    def n_dims(self) -> int:
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

    def __call__(self, coordinates: Tensor) -> Tensor:
        return coordinates

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

    def to(
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
    def n_dims(self) -> int:
        return self._n_dims


class CPUAffineTransformation(BaseAffineTransformation):
    """Affine tranformation for which compositions and inversions are done
    actively on CPU, and the same computations on target devices are done only
    if needed

    Allows to do control flow decisions on CPU based on the actively computed
    transformation matrix without having to do CPU-GPU synchronization.

    Arguments:
        transformation_matrix_on_cpu: Transformation matrix on cpu
        device: Device to use for the transformation matrix produced by as_matrix method
        pin_memory: Whether to pin memory for the cpu transformation matrix
    """

    def __init__(
        self,
        transformation_matrix_on_cpu: Tensor,
        device: Optional[torch_device] = None,
        pin_memory: bool = True,
    ) -> None:
        if transformation_matrix_on_cpu.device != torch_device("cpu"):
            raise ValueError("Please give the matrix on CPU")
        if transformation_matrix_on_cpu.requires_grad:
            raise ValueError("The implementation assumes a detached transformation matrix.")
        self._transformation_matrix_cpu = (
            transformation_matrix_on_cpu.pin_memory()
            if pin_memory
            else transformation_matrix_on_cpu
        )
        self._device = torch_device("cpu") if device is None else device

    def __call__(self, coordinates: Tensor) -> Tensor:
        if self._is_identity():
            return coordinates
        return _broadcast_and_transform_coordinates(coordinates, self.as_matrix())

    def as_matrix(
        self,
    ) -> Tensor:
        return self._transformation_matrix_cpu.to(
            device=self._device, non_blocking=self._device != torch_device("cpu")
        )

    def as_cpu_matrix(self) -> Tensor:
        return self._transformation_matrix_cpu

    def _is_identity(self) -> bool:
        first_channel_dim = index_by_channel_dims(
            n_total_dims=self._transformation_matrix_cpu.ndim, channel_dim_index=0, n_channel_dims=2
        )
        n_rows = self._transformation_matrix_cpu.size(first_channel_dim)
        identity_matrix = eye(
            n_rows,
            dtype=self._transformation_matrix_cpu.dtype,
            device=self._transformation_matrix_cpu.device,
        )
        broadcasted_identity_matrix = broadcast_to_shape_around_channel_dims(
            identity_matrix, shape=self._transformation_matrix_cpu.shape, n_channel_dims=2
        )
        return allclose(
            self._transformation_matrix_cpu,
            broadcasted_identity_matrix,
        )

    def invert(self) -> "CPUAffineTransformation":
        return _CPUAffineTransformationInverse(
            inverted_transformation_matrix_cpu=channels_last(2, 2)(inverse)(
                self._transformation_matrix_cpu
            ),
            transformation_to_invert=self,
        )

    def detach(self) -> "CPUAffineTransformation":
        return self

    def compose(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if isinstance(affine_transformation, CPUAffineTransformation):
            return _CPUAffineTransformationComposition(
                compsed_transformation_matrix_cpu=compose_affine_transformation_matrices(
                    self._transformation_matrix_cpu, affine_transformation.as_cpu_matrix()
                ),
                left_transformation=self,
                right_transformation=affine_transformation,
            )
        return super().compose(affine_transformation)

    def reduce(self, pin_memory: bool = True) -> "CPUAffineTransformation":
        """Reduce the transformation to non-lazy version"""
        return CPUAffineTransformation(
            self.as_cpu_matrix(), device=self._device, pin_memory=pin_memory
        )

    @property
    def n_dims(self) -> int:
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

    def to(
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


class _CPUAffineTransformationInverse(CPUAffineTransformation):
    def __init__(
        self,
        inverted_transformation_matrix_cpu: Tensor,
        transformation_to_invert: CPUAffineTransformation,
    ) -> None:
        super().__init__(
            inverted_transformation_matrix_cpu,
            device=transformation_to_invert.device,
            pin_memory=False,
        )
        self._transformation_to_invert = transformation_to_invert

    def to(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "_CPUAffineTransformationInverse":
        return _CPUAffineTransformationInverse(
            inverted_transformation_matrix_cpu=self._transformation_matrix_cpu.to(
                dtype=self._transformation_matrix_cpu.dtype if dtype is None else dtype,
            ),
            transformation_to_invert=self._transformation_to_invert.to(dtype=dtype, device=device),
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return channels_last(2, 2)(inverse)(self._transformation_to_invert.as_matrix())


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
            pin_memory=False,
        )
        self._left_transformation = left_transformation
        self._right_transformation = right_transformation

    def to(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "_CPUAffineTransformationComposition":
        return _CPUAffineTransformationComposition(
            compsed_transformation_matrix_cpu=self._transformation_matrix_cpu.to(
                dtype=self._transformation_matrix_cpu.dtype if dtype is None else dtype,
            ),
            left_transformation=self._left_transformation.to(dtype=dtype, device=device),
            right_transformation=self._right_transformation.to(dtype=dtype, device=device),
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return compose_affine_transformation_matrices(
            self._left_transformation.as_matrix(), self._right_transformation.as_matrix()
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
    raise RuntimeError("Could not infer affine transformation")


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
    ):
        raise RuntimeError(
            "Affine tracer has no values or mask! Usually this error means that "
            "the traced mapping is not affine."
        )

    def generate_mask(
        self,
        generate_missing_mask: bool = True,
    ):
        raise RuntimeError(
            "Affine tracer has no mask! Usually this error means that "
            "the traced mapping is not affine."
        )

    def generate_values(
        self,
    ) -> Tensor:
        raise RuntimeError(
            "Affine tracer has no values! Usually this error means that "
            "the traced mapping is not affine."
        )

    def apply_affine(self, affine_transformation: IAffineTransformation) -> "IMaskedTensor":
        return _AffineTracer(affine_transformation.compose(self.affine_transformation))

    def has_mask(self) -> bool:
        return False

    @property
    def channels_shape(self) -> Sequence[int]:
        raise RuntimeError(
            "Affine tracer has no channels! Usually this error means that "
            "the traced mapping is not affine."
        )

    @property
    def shape(self) -> Sequence[int]:
        raise RuntimeError(
            "Affine tracer has no shape! Usually this error means that "
            "the traced mapping is not affine."
        )

    def clear_mask(self) -> "_AffineTracer":
        return self

    def detach(self) -> "_AffineTracer":
        return self

    def as_slice(self, target_shape: Sequence[int]) -> None:
        return None

    def reduce(self) -> IMaskedTensor:
        return _AffineTracer(IdentityAffineTransformation(self.affine_transformation.n_dims))


def compose_affine_transformation_matrices(
    transformation_1: Tensor, transformation_2: Tensor
) -> Tensor:
    """Compose two transformation matrices

    Args:
        transformation_1: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, *)
        transformation_2: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, *)

    Returns: transformation_1: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, *)
    """
    transformation_1 = move_channels_last(transformation_1, 2)
    transformation_2 = move_channels_last(transformation_2, 2)
    composed = matmul(transformation_1, transformation_2)
    return move_channels_first(composed, 2)


def convert_to_homogenous_coordinates(coordinates: Tensor) -> Tensor:
    """Converts the coordinates to homogenous coordinates

    Args:
        coordinates: Tensor with shape (batch_size, n_channels, *)
        channels_first: Whether to have channels first, default True

    Returns: Tensor with shape (batch_size, n_channels + 1, *)
    """
    coordinates = move_channels_last(coordinates)
    coordinates, batch_dimensions_shape = merge_batch_dimensions(coordinates)
    homogenous_coordinates = cat(
        [
            coordinates,
            ones(1, device=coordinates.device, dtype=coordinates.dtype).expand(
                coordinates.size(0), 1
            ),
        ],
        dim=-1,
    )
    homogenous_coordinates = unmerge_batch_dimensions(
        homogenous_coordinates, batch_dimensions_shape=batch_dimensions_shape
    )
    return move_channels_first(homogenous_coordinates)


@channels_last({"coordinates": 1, "transformation_matrix": 2}, 1)
@merged_batch_dimensions({"coordinates": 1, "transformation_matrix": 2}, 1)
def _transform_coordinates(coordinates: Tensor, transformation_matrix: Tensor) -> Tensor:
    transformed = matmul(
        transformation_matrix, convert_to_homogenous_coordinates(coordinates)[..., None]
    )[..., :-1, 0]
    return transformed


def _broadcast_and_transform_coordinates(
    coordinates: Tensor, transformation_matrix: Tensor
) -> Tensor:
    coordinates, transformation_matrix = broadcast_tensors_around_channel_dims(
        (coordinates, transformation_matrix), n_channel_dims=(1, 2)
    )
    return _transform_coordinates(coordinates, transformation_matrix)
