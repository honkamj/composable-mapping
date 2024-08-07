"""Affine transformation implementations"""

from typing import List, Mapping, Optional, Sequence

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

    def __repr__(self) -> str:
        return (
            f"IdentityAffineTransformation("
            f"n_dims={self._n_dims}, "
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

    def __call__(self, coordinates: Tensor) -> Tensor:
        if self._is_identity():
            return coordinates
        return _broadcast_and_transform_coordinates(coordinates, self.as_matrix())

    def pin_memory_if_needed(self) -> "CPUAffineTransformation":
        """Pin memory of the cpu transformation matrices needed for producing
        output of as_matrix on gpu"""
        return CPUAffineTransformation(
            (
                self._transformation_matrix_cpu.pin_memory()
                if self._device != torch_device("cpu")
                else self._transformation_matrix_cpu
            ),
            device=self._device,
        )

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

    def reduce(self) -> "CPUAffineTransformation":
        """Reduce the transformation to non-lazy version"""
        return CPUAffineTransformation(
            self.as_cpu_matrix(),
            device=self._device,
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

    def pin_memory_if_needed(self) -> "_CPUAffineTransformationInverse":
        return _CPUAffineTransformationInverse(
            inverted_transformation_matrix_cpu=self.as_cpu_matrix(),
            transformation_to_invert=(self._transformation_to_invert.pin_memory_if_needed()),
        )

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

    def pin_memory_if_needed(self) -> "_CPUAffineTransformationComposition":
        return _CPUAffineTransformationComposition(
            compsed_transformation_matrix_cpu=self.as_cpu_matrix(),
            left_transformation=self._left_transformation.pin_memory_if_needed(),
            right_transformation=self._right_transformation.pin_memory_if_needed(),
        )

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

    def __repr__(self) -> str:
        return (
            f"_CPUAffineTransformationComposition("
            f"compsed_transformation_matrix_cpu={self._transformation_matrix_cpu}, "
            f"left_transformation={self._left_transformation}, "
            f"right_transformation={self._right_transformation})"
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
    ):
        raise NotAffineTransformationError(
            "Affine tracer has no values or mask! Usually this error means that "
            "the traced mapping is not affine."
        )

    def generate_mask(
        self,
        generate_missing_mask: bool = True,
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

    def apply_affine(self, affine_transformation: IAffineTransformation) -> "IMaskedTensor":
        return _AffineTracer(affine_transformation.compose(self.affine_transformation))

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
        return _AffineTracer(IdentityAffineTransformation(self.affine_transformation.n_dims))

    def __repr__(self) -> str:
        return f"_AffineTracer(affine_transformation={self.affine_transformation})"

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
        embedded_matrix, batch_dimensions_shape=batch_dimensions_shape, num_channel_dims=2
    )
    return move_channels_first(embedded_matrix, 2)


@script
def generate_translation_matrix(translations: Tensor) -> Tensor:
    """Generator homogenous translation matrix with given translations

    Args:
        translations: Tensor with shape (batch_size, n_dims, ...)

    Returns: Tensor with shape (batch_size, n_dims + 1, n_dims + 1, ...)
    """
    translations = move_channels_last(translations)
    translations, batch_dimensions_shape = merge_batch_dimensions(translations)
    batch_size = translations.size(0)
    n_dims = translations.size(1)
    homogenous_translation = convert_to_homogenous_coordinates(coordinates=translations)
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
        translation_matrix, batch_dimensions_shape=batch_dimensions_shape, num_channel_dims=2
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
    scales = move_channels_last(scales)
    scale_matrix = diag_embed(scales)
    return move_channels_first(scale_matrix, num_channel_dims=2)


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
