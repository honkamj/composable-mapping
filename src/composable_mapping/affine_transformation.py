"""Affine transformation implementations"""

from abc import abstractmethod
from typing import Mapping, Optional, Sequence, Tuple

from torch import Tensor, allclose, cat
from torch import device as torch_device
from torch import diag_embed
from torch import dtype as torch_dtype
from torch import eye, inverse, matmul, ones, zeros

from .tensor_like import BaseTensorLikeWrapper, ITensorLike
from .util import (
    broadcast_shapes_in_parts,
    broadcast_tensors_in_parts,
    broadcast_to_in_parts,
    get_channel_dims,
    move_channels_first,
    move_channels_last,
    split_shape,
)

IDENTITY_MATRIX_TOLERANCE = 1e-5
ZERO_MATRIX_TOLERANCE = 1e-5


class IAffineTransformation(ITensorLike):
    """Affine transformation"""

    @abstractmethod
    def __matmul__(self, affine_transformation: "IAffineTransformation") -> "IAffineTransformation":
        """Compose with another affine transformation"""

    @abstractmethod
    def __add__(self, affine_transformation: "IAffineTransformation") -> "IAffineTransformation":
        """Add another affine transformation"""

    @abstractmethod
    def __sub__(self, affine_transformation: "IAffineTransformation") -> "IAffineTransformation":
        """Subtract another affine transformation"""

    @abstractmethod
    def __neg__(self) -> "IAffineTransformation":
        """Negate the affine transformation"""

    @abstractmethod
    def as_matrix(
        self,
    ) -> Tensor:
        """Return the mapping as matrix"""

    @abstractmethod
    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        """Evaluate the mapping with values"""

    @abstractmethod
    def invert(self) -> "IAffineTransformation":
        """Invert the transformation"""

    @abstractmethod
    def as_cpu_matrix(self) -> Optional[Tensor]:
        """Returns the transformation matrix detached on cpu, if available"""

    @abstractmethod
    def get_output_shape(
        self, input_shape: Sequence[int], n_channel_dims: int = 1
    ) -> Tuple[int, ...]:
        """Return the shape of the output tensor given the input shape

        Raises an error if the transformation is not compatible with the input shape
        """

    @abstractmethod
    def is_zero(self) -> Optional[bool]:
        """Return whether the transformation is zero

        Returns None if the check cannot be done on CPU

        Args:
            n_input_dims: Number of dimensions the transformation is applied on
            n_output_dims: Number of dimensions the transformation outputs
        """


class BaseAffineTransformation(IAffineTransformation):
    """Base affine transformation"""

    @property
    @abstractmethod
    def _matrix_shape(self) -> Sequence[int]:
        """Return shape of the transformation matrix"""

    def get_output_shape(
        self, input_shape: Sequence[int], n_channel_dims: int = 1
    ) -> Tuple[int, ...]:
        broadcasted_input_shape, _broadcasted_matrix_shape = broadcast_shapes_in_parts(
            input_shape,
            self._matrix_shape,
            n_channel_dims=(n_channel_dims, 2),
            broadcast_channels=False,
        )
        matrix_channel_dims = get_channel_dims(len(self._matrix_shape), n_channel_dims=2)
        n_matrix_input_channels = self._matrix_shape[matrix_channel_dims[1]] - 1
        last_input_channel_index = get_channel_dims(
            len(input_shape), n_channel_dims=n_channel_dims
        )[-1]
        if input_shape[last_input_channel_index] != n_matrix_input_channels:
            raise ValueError("Input shape does not match the transformation matrix")
        n_output_channels = self._matrix_shape[matrix_channel_dims[0]] - 1
        modified_input_shape = list(broadcasted_input_shape)
        modified_input_shape[last_input_channel_index] = n_output_channels
        return tuple(modified_input_shape)

    def __matmul__(self, affine_transformation: IAffineTransformation) -> "IAffineTransformation":
        self_cpu_matrix = self.as_cpu_matrix()
        target_cpu_matrix = affine_transformation.as_cpu_matrix()
        if self_cpu_matrix is not None and target_cpu_matrix is not None:
            return CPUAffineTransformation(
                transformation_matrix_on_cpu=compose_affine_matrices(
                    self_cpu_matrix, target_cpu_matrix
                ),
                device=self.device,
            )
        return AffineTransformation(
            compose_affine_matrices(
                self.as_matrix(),
                affine_transformation.as_matrix(),
            )
        )

    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        self_cpu_matrix = self.as_cpu_matrix()
        target_cpu_matrix = affine_transformation.as_cpu_matrix()
        if self_cpu_matrix is not None and target_cpu_matrix is not None:
            return CPUAffineTransformation(
                transformation_matrix_on_cpu=add_affine_matrices(
                    self_cpu_matrix, target_cpu_matrix
                ),
                device=self.device,
            )
        return AffineTransformation(
            add_affine_matrices(self.as_matrix(), affine_transformation.as_matrix())
        )

    def __sub__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        self_cpu_matrix = self.as_cpu_matrix()
        target_cpu_matrix = affine_transformation.as_cpu_matrix()
        if self_cpu_matrix is not None and target_cpu_matrix is not None:
            return CPUAffineTransformation(
                transformation_matrix_on_cpu=substract_affine_matrices(
                    self_cpu_matrix, target_cpu_matrix
                ),
                device=self.device,
            )
        return AffineTransformation(
            substract_affine_matrices(self.as_matrix(), affine_transformation.as_matrix())
        )

    def is_zero(self) -> Optional[bool]:
        cpu_matrix = self.as_cpu_matrix()
        if cpu_matrix is not None:
            return is_zero_matrix(cpu_matrix)
        return None


class AffineTransformation(BaseAffineTransformation, BaseTensorLikeWrapper):
    """Represents generic affine transformation

    Arguments:
        transformation_matrix: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, ...),
            if None, corresponds to identity transformation
    """

    def __init__(self, transformation_matrix: Tensor) -> None:
        self._transformation_matrix = transformation_matrix

    @property
    def _matrix_shape(self) -> Sequence[int]:
        return self._transformation_matrix.shape

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {"transformation_matrix": self._transformation_matrix}

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "AffineTransformation":
        return AffineTransformation(tensors["transformation_matrix"])

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        return transform_values_with_affine_matrix(
            self._transformation_matrix,
            values,
            n_channel_dims=n_channel_dims,
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return self._transformation_matrix

    def __neg__(self) -> IAffineTransformation:
        return AffineTransformation(negate_affine_matrix(self.as_matrix()))

    def invert(self) -> IAffineTransformation:
        return AffineTransformation(invert_matrix(self.as_matrix()))

    def as_cpu_matrix(self) -> None:
        return None

    def __repr__(self) -> str:
        return f"AffineTransformation(transformation_matrix={self._transformation_matrix})"


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
        self._transformation_matrix_on_cpu = transformation_matrix_on_cpu
        self._device = torch_device("cpu") if device is None else device
        if self._device != torch_device("cpu"):
            self._transformation_matrix_target_device = (
                self._transformation_matrix_on_cpu.pin_memory().to(
                    device=self._device, non_blocking=True
                )
            )
        else:
            self._transformation_matrix_target_device = transformation_matrix_on_cpu

    @property
    def _matrix_shape(self) -> Sequence[int]:
        return self._transformation_matrix_on_cpu.shape

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        if (
            is_identity_matrix(self._transformation_matrix_on_cpu)
            and self.get_output_shape(values.shape) == values.shape
        ):
            return values
        if self.is_zero():
            return zeros(
                self.get_output_shape(values.shape, n_channel_dims),
                device=self._device,
                dtype=values.dtype,
            )
        return transform_values_with_affine_matrix(
            self.as_matrix(), values, n_channel_dims=n_channel_dims
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return self._transformation_matrix_on_cpu.to(
            device=self._device, non_blocking=self._device != torch_device("cpu")
        )

    def as_cpu_matrix(self) -> Tensor:
        return self._transformation_matrix_on_cpu

    def detach(self) -> "CPUAffineTransformation":
        return self

    def reduce(self) -> "CPUAffineTransformation":
        """Reduce the transformation to non-lazy version"""
        return CPUAffineTransformation(
            self.as_cpu_matrix(),
            device=self._device,
        )

    @property
    def dtype(
        self,
    ) -> torch_dtype:
        return self._transformation_matrix_on_cpu.dtype

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
            transformation_matrix_on_cpu=self._transformation_matrix_on_cpu.to(
                dtype=self._transformation_matrix_on_cpu.dtype if dtype is None else dtype,
            ),
            device=self._device if device is None else device,
        )

    def __repr__(self) -> str:
        return (
            f"CPUAffineTransformation("
            f"transformation_matrix_on_cpu={self._transformation_matrix_on_cpu}, "
            f"device={self._device})"
        )

    def __neg__(self) -> "CPUAffineTransformation":
        return CPUAffineTransformation(
            transformation_matrix_on_cpu=negate_affine_matrix(self.as_cpu_matrix()),
            device=self.device,
        )

    def invert(self) -> "CPUAffineTransformation":
        return CPUAffineTransformation(
            transformation_matrix_on_cpu=invert_matrix(self.as_cpu_matrix()),
            device=self.device,
        )


def compose_affine_matrices(
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


def embed_matrix(matrix: Tensor, target_shape: Sequence[int]) -> Tensor:
    """Embed transformation into larger dimensional space

    Args:
        matrix: Tensor with shape ([batch_size, ]n_dims, n_dims, ...)
        target_shape: Target matrix shape

    Returns: Tensor with shape (batch_size, *target_shape)
    """
    if len(target_shape) != 2:
        raise ValueError("Matrix shape must be two dimensional.")
    batch_dimensions_shape, channel_shape, spatial_shape = split_shape(
        matrix.shape, n_channel_dims=2
    )
    n_rows_needed = target_shape[0] - channel_shape[0]
    n_cols_needed = target_shape[1] - channel_shape[1]
    if n_rows_needed == 0 and n_cols_needed == 0:
        return matrix
    rows = cat(
        [
            zeros(
                n_rows_needed,
                min(channel_shape[1], channel_shape[0]),
                device=matrix.device,
                dtype=matrix.dtype,
            ),
            eye(
                n_rows_needed,
                max(0, channel_shape[1] - channel_shape[0]),
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ],
        dim=1,
    ).expand(*batch_dimensions_shape, -1, -1, *spatial_shape)
    cols = cat(
        [
            zeros(
                min(target_shape[0], channel_shape[1]),
                n_cols_needed,
                device=matrix.device,
                dtype=matrix.dtype,
            ),
            eye(
                max(0, target_shape[0] - channel_shape[1]),
                n_cols_needed,
                device=matrix.device,
                dtype=matrix.dtype,
            ),
        ],
        dim=0,
    ).expand(*batch_dimensions_shape, -1, -1, *spatial_shape)
    channel_dims = get_channel_dims(matrix.ndim, n_channel_dims=2)
    embedded_matrix = cat([cat([matrix, rows], dim=channel_dims[0]), cols], dim=channel_dims[1])
    return embedded_matrix


def convert_to_homogenous_coordinates(coordinates: Tensor, dim: int = -1) -> Tensor:
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
    batch_dimensions_shape, _, spatial_shape = split_shape(translations.shape, n_channel_dims=1)
    channel_dim = get_channel_dims(translations.ndim, n_channel_dims=1)[0]
    n_dims = translations.size(channel_dim)
    homogenous_translation = convert_to_homogenous_coordinates(translations, dim=-1)
    translation_matrix = cat(
        [
            cat(
                [
                    eye(n_dims, device=translations.device, dtype=translations.dtype),
                    zeros(1, n_dims, device=translations.device, dtype=translations.dtype),
                ],
                dim=0,
            ).expand(*batch_dimensions_shape, -1, -1, *spatial_shape),
            homogenous_translation.unsqueeze(channel_dim + 1),
        ],
        dim=channel_dim + 1,
    )
    return translation_matrix


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


def invert_matrix(matrix: Tensor) -> Tensor:
    """Invert a matrix or a batch of matrices"""
    matrix = move_channels_last(matrix, 2)
    inverted_matrix = inverse(matrix)
    return move_channels_first(inverted_matrix, 2)


def add_affine_matrices(matrix_1: Tensor, matrix_2: Tensor) -> Tensor:
    """Add two affine matrices or batches of matrices

    The last row of the matrices is not included in the addition
    and is copied from the first matrix, as it should always be
    [0, ..., 0, 1] for affine transformations.
    """
    matrix_1, matrix_2 = broadcast_tensors_in_parts(
        matrix_1, matrix_2, broadcast_channels=False, n_channel_dims=2
    )
    if matrix_1.shape != matrix_2.shape:
        raise ValueError("Matrices are not broadcastable.")
    matrix_1 = move_channels_last(matrix_1, 2)
    matrix_2 = move_channels_last(matrix_2, 2)
    sum_matrix = matrix_1[..., :-1, :] + matrix_2[..., :-1, :]
    sum_matrix = cat(
        [
            sum_matrix,
            matrix_1[..., -1:, :],
        ],
        dim=-2,
    )
    sum_matrix = move_channels_first(sum_matrix, 2)
    return sum_matrix


def substract_affine_matrices(matrix_1: Tensor, matrix_2: Tensor) -> Tensor:
    """Subtract two affine matrices or batches of matrices

    The last row of the matrices is not included in the subtraction
    and is copied from the first matrix, as it should always be
    [0, ..., 0, 1] for affine transformations.
    """
    matrix_1, matrix_2 = broadcast_tensors_in_parts(
        matrix_1, matrix_2, broadcast_channels=False, n_channel_dims=2
    )
    if matrix_1.shape != matrix_2.shape:
        raise ValueError("Matrices are not broadcastable.")
    matrix_1 = move_channels_last(matrix_1, 2)
    matrix_2 = move_channels_last(matrix_2, 2)
    diff_matrix = matrix_1[..., :-1, :] - matrix_2[..., :-1, :]
    diff_matrix = cat(
        [
            diff_matrix,
            matrix_1[..., -1:, :],
        ],
        dim=-2,
    )
    diff_matrix = move_channels_first(diff_matrix, 2)
    return diff_matrix


def negate_affine_matrix(matrix: Tensor) -> Tensor:
    """Negate an affine matrix or a batch of matrices

    The last row of the matrix is not negated as it should always be
    [0, ..., 0, 1] for affine transformations.
    """
    matrix = move_channels_last(matrix, 2)
    negated_matrix = -matrix[..., :-1, :]
    negated_matrix = cat(
        [
            negated_matrix,
            matrix[..., -1:, :],
        ],
        dim=-2,
    )
    negated_matrix = move_channels_first(negated_matrix, 2)
    return negated_matrix


def is_zero_matrix(matrix: Tensor) -> bool:
    """Return whether a matrix or batch of matrices is a zero matrix"""
    row_dimension = get_channel_dims(matrix.ndim, n_channel_dims=2)[0]
    return allclose(
        matrix.moveaxis(row_dimension, -1)[..., :-1],
        zeros(
            1,
            dtype=matrix.dtype,
            device=matrix.device,
        ),
        atol=ZERO_MATRIX_TOLERANCE,
    )


def is_identity_matrix(matrix: Tensor) -> bool:
    """Return whether a matrix or batch of matrices is an identity"""
    if matrix.size(-2) != matrix.size(-1):
        return False
    n_rows = matrix.size(get_channel_dims(matrix.ndim, n_channel_dims=2)[0])
    identity_matrix = eye(
        n_rows,
        dtype=matrix.dtype,
        device=matrix.device,
    )
    batch_shape, _, spatial_shape = split_shape(matrix.shape, n_channel_dims=2)
    broadcasted_identity_matrix = broadcast_to_in_parts(
        identity_matrix, batch_shape=batch_shape, spatial_shape=spatial_shape, n_channel_dims=2
    )
    return broadcasted_identity_matrix.shape == matrix.shape and allclose(
        matrix, broadcasted_identity_matrix, atol=IDENTITY_MATRIX_TOLERANCE
    )


def transform_values_with_affine_matrix(
    transformation_matrix: Tensor, values: Tensor, n_channel_dims: int = 1
) -> Tensor:
    """Transform values with affine matrix"""
    values, transformation_matrix = broadcast_tensors_in_parts(
        values, transformation_matrix, broadcast_channels=False, n_channel_dims=(n_channel_dims, 2)
    )
    transformation_matrix = move_channels_last(transformation_matrix, 2)
    if n_channel_dims > 2:
        transformation_matrix = transformation_matrix[
            (...,) + (None,) * (n_channel_dims - 2) + 2 * (slice(None),)
        ]
    values = move_channels_last(values, n_channel_dims)
    transformed = matmul(
        transformation_matrix,
        convert_to_homogenous_coordinates(values, dim=-1)[..., None],
    )[..., :-1, 0]
    transformed = move_channels_first(transformed, n_channel_dims)

    return transformed
