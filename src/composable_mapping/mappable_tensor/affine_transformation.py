"""Affine transformation implementations"""

from abc import abstractmethod
from typing import Mapping, Optional, Sequence, Tuple, Union

from torch import Tensor, allclose, broadcast_shapes, cat
from torch import device as torch_device
from torch import diag_embed
from torch import dtype as torch_dtype
from torch import eye, get_default_dtype, inverse, matmul, ones, zeros

from composable_mapping.tensor_like import BaseTensorLikeWrapper, ITensorLike
from composable_mapping.util import (
    broadcast_optional_shapes_in_parts_splitted,
    broadcast_shapes_in_parts,
    broadcast_shapes_in_parts_splitted,
    broadcast_tensors_in_parts,
    broadcast_to_in_parts,
    get_channel_dims,
    get_channels_shape,
    move_channels_first,
    move_channels_last,
    split_shape,
)

IDENTITY_MATRIX_TOLERANCE = 1e-5
ZERO_MATRIX_TOLERANCE = 1e-5


HostAffineTransformationType = Union["HostAffineTransformation", "HostDiagonalAffineTransformation"]


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

    @property
    @abstractmethod
    def matrix_shape(self) -> Sequence[int]:
        """Return shape of the transformation matrix"""


class BaseAffineTransformation(IAffineTransformation):
    """Base affine transformation"""

    @property
    @abstractmethod
    def matrix_shape(self) -> Sequence[int]:
        """Return shape of the transformation matrix"""

    def get_output_shape(
        self, input_shape: Sequence[int], n_channel_dims: int = 1
    ) -> Tuple[int, ...]:
        broadcasted_input_shape, _broadcasted_matrix_shape = broadcast_shapes_in_parts(
            input_shape,
            self.matrix_shape,
            n_channel_dims=(n_channel_dims, 2),
            broadcast_channels=False,
        )
        matrix_channel_dims = get_channel_dims(len(self.matrix_shape), n_channel_dims=2)
        n_matrix_input_channels = self.matrix_shape[matrix_channel_dims[1]] - 1
        last_input_channel_index = get_channel_dims(
            len(input_shape), n_channel_dims=n_channel_dims
        )[-1]
        if input_shape[last_input_channel_index] != n_matrix_input_channels:
            raise ValueError("Input shape does not match the transformation matrix")
        n_output_channels = self.matrix_shape[matrix_channel_dims[0]] - 1
        modified_input_shape = list(broadcasted_input_shape)
        modified_input_shape[last_input_channel_index] = n_output_channels
        return tuple(modified_input_shape)

    def __matmul__(self, affine_transformation: IAffineTransformation) -> "IAffineTransformation":
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        self_cpu_matrix = self.as_cpu_matrix()
        target_cpu_matrix = affine_transformation.as_cpu_matrix()
        if self_cpu_matrix is not None and target_cpu_matrix is not None:
            return HostAffineTransformation(
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
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        self_cpu_matrix = self.as_cpu_matrix()
        target_cpu_matrix = affine_transformation.as_cpu_matrix()
        if self_cpu_matrix is not None and target_cpu_matrix is not None:
            return HostAffineTransformation(
                transformation_matrix_on_cpu=add_affine_matrices(
                    self_cpu_matrix, target_cpu_matrix
                ),
                device=self.device,
            )
        return AffineTransformation(
            add_affine_matrices(self.as_matrix(), affine_transformation.as_matrix())
        )

    def __sub__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        return self.__add__(-affine_transformation)

    def is_zero(self) -> Optional[bool]:
        cpu_matrix = self.as_cpu_matrix()
        if cpu_matrix is not None:
            return is_zero_matrix(cpu_matrix)
        return None


class AffineTransformation(BaseAffineTransformation, BaseTensorLikeWrapper):
    """Represents generic affine transformation

    Arguments:
        transformation_matrix: Tensor with shape ([batch_size, ]n_dims + 1, n_dims + 1, ...)
    """

    def __init__(self, transformation_matrix: Tensor) -> None:
        self._transformation_matrix = transformation_matrix

    @property
    def matrix_shape(self) -> Sequence[int]:
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
            self.as_matrix(),
            values,
            n_channel_dims=n_channel_dims,
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return self._transformation_matrix

    def as_cpu_matrix(self) -> Optional[Tensor]:
        return None

    def __neg__(self) -> IAffineTransformation:
        return AffineTransformation(negate_affine_matrix(self.as_matrix()))

    def invert(self) -> IAffineTransformation:
        return AffineTransformation(invert_matrix(self.as_matrix()))

    def __repr__(self) -> str:
        return f"AffineTransformation(transformation_matrix={self._transformation_matrix})"


class HostAffineTransformation(AffineTransformation):
    """Affine tranformation for which matrix operations are done
    on host, and the matrix on target device is created only when needed

    Allows to do control flow decisions on host without having to do CPU-GPU
    synchronization.

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
        super().__init__(transformation_matrix=transformation_matrix_on_cpu)
        self._device = torch_device("cpu") if device is None else device

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        if (
            is_identity_matrix(self._transformation_matrix)
            and self.get_output_shape(values.shape) == values.shape
        ):
            return values
        if self.is_zero():
            return zeros(
                self.get_output_shape(values.shape, n_channel_dims),
                device=self._device,
                dtype=values.dtype,
            )
        return super().__call__(values, n_channel_dims)

    def as_matrix(
        self,
    ) -> Tensor:
        matrix = super().as_matrix()
        if self._device != torch_device("cpu"):
            matrix = matrix.pin_memory().to(device=self._device, non_blocking=True)
        return matrix

    def as_cpu_matrix(self) -> Tensor:
        return super().as_matrix()

    def detach(self) -> "HostAffineTransformation":
        return self

    @property
    def device(
        self,
    ) -> torch_device:
        return self._device

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "HostAffineTransformation":
        return HostAffineTransformation(
            transformation_matrix_on_cpu=self._transformation_matrix.to(
                dtype=self._transformation_matrix.dtype if dtype is None else dtype,
            ),
            device=self._device if device is None else device,
        )

    def __repr__(self) -> str:
        return (
            f"HostAffineTransformation("
            f"transformation_matrix_on_cpu={self._transformation_matrix}, "
            f"device={self._device})"
        )

    def __neg__(self) -> "HostAffineTransformation":
        return HostAffineTransformation(
            transformation_matrix_on_cpu=negate_affine_matrix(self.as_cpu_matrix()),
            device=self.device,
        )

    def invert(self) -> "HostAffineTransformation":
        return HostAffineTransformation(
            transformation_matrix_on_cpu=invert_matrix(self.as_cpu_matrix()),
            device=self.device,
        )


class BaseDiagonalAffineTransformation(BaseAffineTransformation):
    """Base diagonal affine transformation"""

    def to(
        self, dtype: Optional[torch_dtype] = None, device: Optional[torch_device] = None
    ) -> "IAffineTransformation":
        """Alias for cast method following the torch convention"""
        return self.cast(dtype=dtype, device=device)

    def __matmul__(self, affine_transformation: IAffineTransformation) -> "IAffineTransformation":
        if isinstance(affine_transformation, BaseDiagonalAffineTransformation):
            self_cpu_diagonal = self.as_cpu_diagonal()
            target_cpu_diagonal = affine_transformation.as_cpu_diagonal()
            if self_cpu_diagonal is not None and target_cpu_diagonal is not None:
                composed_cpu_diagonal, composed_cpu_translation, composed_matrix_shape = (
                    compose_diagonal_affine_matrices(
                        diagonal_1=self_cpu_diagonal[0],
                        translation_1=self_cpu_diagonal[1],
                        matrix_shape_1=self.matrix_shape,
                        diagonal_2=target_cpu_diagonal[0],
                        translation_2=target_cpu_diagonal[1],
                        matrix_shape_2=affine_transformation.matrix_shape,
                    )
                )
                return HostDiagonalAffineTransformation(
                    diagonal=composed_cpu_diagonal,
                    translation=composed_cpu_translation,
                    matrix_shape=composed_matrix_shape,
                    dtype=self.dtype,
                    device=self.device,
                )

            self_diagonal = self.as_diagonal()
            target_diagonal = affine_transformation.as_diagonal()
            composed_diagonal, composed_translation, composed_matrix_shape = (
                compose_diagonal_affine_matrices(
                    diagonal_1=self_diagonal[0],
                    translation_1=self_diagonal[1],
                    matrix_shape_1=self.matrix_shape,
                    diagonal_2=target_diagonal[0],
                    translation_2=target_diagonal[1],
                    matrix_shape_2=affine_transformation.matrix_shape,
                )
            )
            return DiagonalAffineTransformation(
                diagonal=composed_diagonal,
                translation=composed_translation,
                matrix_shape=composed_matrix_shape,
                dtype=self.dtype,
                device=self.device,
            )
        return super().__matmul__(affine_transformation)

    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if isinstance(affine_transformation, BaseDiagonalAffineTransformation):
            self_cpu_diagonal = self.as_cpu_diagonal()
            target_cpu_diagonal = affine_transformation.as_cpu_diagonal()
            if self_cpu_diagonal is not None and target_cpu_diagonal is not None:
                added_cpu_diagonals, added_cpu_translations, added_matrix_shape = (
                    add_diagonal_affine_matrices(
                        diagonal_1=self_cpu_diagonal[0],
                        translation_1=self_cpu_diagonal[1],
                        matrix_shape_1=self.matrix_shape,
                        diagonal_2=target_cpu_diagonal[0],
                        translation_2=target_cpu_diagonal[1],
                        matrix_shape_2=affine_transformation.matrix_shape,
                    )
                )
                return HostDiagonalAffineTransformation(
                    diagonal=added_cpu_diagonals,
                    translation=added_cpu_translations,
                    matrix_shape=added_matrix_shape,
                    dtype=self.dtype,
                    device=self.device,
                )

            self_diagonal = self.as_diagonal()
            target_diagonal = affine_transformation.as_diagonal()
            if self_diagonal is not None and target_diagonal is not None:
                added_diagonals, added_translations, added_matrix_shape = (
                    add_diagonal_affine_matrices(
                        diagonal_1=self_diagonal[0],
                        translation_1=self_diagonal[1],
                        matrix_shape_1=self.matrix_shape,
                        diagonal_2=target_diagonal[0],
                        translation_2=target_diagonal[1],
                        matrix_shape_2=affine_transformation.matrix_shape,
                    )
                )
                return DiagonalAffineTransformation(
                    diagonal=added_diagonals,
                    translation=added_translations,
                    matrix_shape=added_matrix_shape,
                    dtype=self.dtype,
                    device=self.device,
                )
        return super().__add__(affine_transformation)

    def is_zero(self) -> Optional[bool]:
        cpu_diagonal_matrix = self.as_cpu_diagonal()
        if cpu_diagonal_matrix is not None:
            return is_zero_diagonal_affine_matrix(cpu_diagonal_matrix[0], cpu_diagonal_matrix[1])
        return None

    @abstractmethod
    def as_diagonal(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Return the diagonal and translation tensors"""

    @abstractmethod
    def as_cpu_diagonal(self) -> Optional[Tuple[Optional[Tensor], Optional[Tensor]]]:
        """Return the diagonal and translation tensors detached on cpu, if available"""


class DiagonalAffineTransformation(BaseDiagonalAffineTransformation):
    """Represents diagonal affine transformation

    Arguments:
        diagonal: Tensor with shape ([batch_size, ]n_source_dims, ...),
            if None, corresponds to all ones
        translation: Tensor with shape ([batch_size, ]n_target_dims, ...),
            if None, corresponds to all zeros
        matrix_shape: Shape of the target affine transformation matrix
            (n_target_dims + 1, n_source_dims + 1)
    """

    def __init__(
        self,
        diagonal: Optional[Tensor] = None,
        translation: Optional[Tensor] = None,
        matrix_shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> None:
        self._diagonal, self._translation, self._dtype, self._device = (
            _handle_diagonal_input_dtype_and_device(diagonal, translation, dtype, device)
        )
        batch_shape, channels_shape, spatial_shape = _get_diagonal_matrix_shape_splitted(
            self._diagonal, self._translation, matrix_shape
        )
        self._matrix_shape = matrix_shape
        self._broadcasted_matrix_shape = batch_shape + channels_shape + spatial_shape

    @property
    def matrix_shape(self) -> Sequence[int]:
        return self._broadcasted_matrix_shape

    def cast(
        self, dtype: Optional[torch_dtype] = None, device: Optional[torch_device] = None
    ) -> "DiagonalAffineTransformation":
        return DiagonalAffineTransformation(
            diagonal=None if self._diagonal is None else self._diagonal.to(dtype=dtype),
            translation=None if self._translation is None else self._translation.to(dtype=dtype),
            matrix_shape=self._matrix_shape,
            dtype=self._dtype if dtype is None else dtype,
            device=self._device if device is None else device,
        )

    @property
    def device(
        self,
    ) -> torch_device:
        return self._device

    @property
    def dtype(
        self,
    ) -> torch_dtype:
        return self._dtype

    def detach(self) -> "DiagonalAffineTransformation":
        return DiagonalAffineTransformation(
            diagonal=None if self._diagonal is None else self._diagonal.detach(),
            translation=None if self._translation is None else self._translation.detach(),
            matrix_shape=self._matrix_shape,
            dtype=self._dtype,
            device=self._device,
        )

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        diagonal, translation = self.as_diagonal()
        return transform_values_with_diagonal_affine_matrix(
            diagonal,
            translation,
            values,
            n_channel_dims=n_channel_dims,
            matrix_shape=self._matrix_shape,
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return generate_diagonal_affine_matrix(
            self._diagonal,
            self._translation,
            matrix_shape=self._matrix_shape,
            dtype=self._dtype,
            device=self._device,
        )

    def as_cpu_matrix(self) -> Optional[Tensor]:
        return None

    def as_diagonal(
        self,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        batch_shape, _, spatial_shape = split_shape(
            self._broadcasted_matrix_shape, n_channel_dims=2
        )
        diagonal = (
            None
            if self._diagonal is None
            else broadcast_to_in_parts(
                self._diagonal,
                batch_shape=batch_shape,
                spatial_shape=spatial_shape,
                n_channel_dims=1,
            )
        )
        translation = (
            None
            if self._translation is None
            else broadcast_to_in_parts(
                self._translation,
                batch_shape=batch_shape,
                spatial_shape=spatial_shape,
                n_channel_dims=1,
            )
        )
        return (diagonal, translation)

    def as_cpu_diagonal(self) -> Optional[Tuple[Optional[Tensor], Optional[Tensor]]]:
        return None

    def __neg__(self) -> IAffineTransformation:
        negated_diagonal, negated_translation = negate_diagonal_affine_matrix(
            self._diagonal,
            self._translation,
            matrix_shape=self._matrix_shape,
            dtype=self._dtype,
            device=self._device,
        )
        return DiagonalAffineTransformation(
            negated_diagonal,
            negated_translation,
            matrix_shape=self._matrix_shape,
            dtype=self._dtype,
            device=self._device,
        )

    def invert(self) -> IAffineTransformation:
        diagonal, translation = invert_diagonal_affine_matrix(
            self._diagonal, self._translation, matrix_shape=self._matrix_shape
        )
        return DiagonalAffineTransformation(
            diagonal,
            translation,
            self._matrix_shape,
        )

    def __repr__(self) -> str:
        return (
            f"DiagonalAffineTransformation(diagonal={self._diagonal}, "
            "translation={self._translation}, matrix_shape={self._matrix_shape})"
        )


class HostDiagonalAffineTransformation(DiagonalAffineTransformation):
    """Diagonal affine tranformation for which matrix operations are done
    on host, and the matrix on target device is created only when needed

    Allows to do control flow decisions on host without having to do CPU-GPU
    synchronization.
    """

    def __init__(
        self,
        diagonal: Optional[Tensor] = None,
        translation: Optional[Tensor] = None,
        matrix_shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> None:
        if diagonal is not None:
            if diagonal.device != torch_device("cpu"):
                raise ValueError("Please give the diagonal on CPU")
            if diagonal.requires_grad:
                raise ValueError("The implementation assumes detached tensors.")
        if translation is not None:
            if translation.device != torch_device("cpu"):
                raise ValueError("Please give the translation on CPU")
            if translation.requires_grad:
                raise ValueError("The implementation assumes detached tensors.")
        super().__init__(
            diagonal=diagonal,
            translation=translation,
            matrix_shape=matrix_shape,
            dtype=dtype,
            device=torch_device("cpu"),
        )
        self._target_device = torch_device("cpu") if device is None else device

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        if (
            is_identity_diagonal_affine_matrix(self._diagonal, self._translation)
            and self.get_output_shape(values.shape) == values.shape
        ):
            return values
        if self.is_zero():
            return zeros(
                self.get_output_shape(values.shape, n_channel_dims),
                device=self._device,
                dtype=values.dtype,
            )
        return super().__call__(values, n_channel_dims)

    def as_matrix(
        self,
    ) -> Tensor:
        matrix = super().as_matrix()
        if self._target_device != torch_device("cpu"):
            matrix = matrix.pin_memory().to(device=self._target_device, non_blocking=True)
        return matrix

    def as_cpu_matrix(self) -> Tensor:
        return super().as_matrix()

    def as_diagonal(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        diagonal, translation = super().as_diagonal()
        if self._target_device != torch_device("cpu"):
            if diagonal is not None:
                diagonal = diagonal.pin_memory().to(device=self._target_device, non_blocking=True)
            if translation is not None:
                translation = translation.pin_memory().to(
                    device=self._target_device, non_blocking=True
                )
        return (
            diagonal,
            translation,
        )

    def as_cpu_diagonal(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        return super().as_diagonal()

    def detach(self) -> "HostDiagonalAffineTransformation":
        return self

    @property
    def device(
        self,
    ) -> torch_device:
        return self._target_device

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "HostDiagonalAffineTransformation":
        dtype = self._dtype if dtype is None else dtype
        device = self._target_device if device is None else device
        return HostDiagonalAffineTransformation(
            diagonal=None if self._diagonal is None else self._diagonal.to(dtype=dtype),
            translation=None if self._translation is None else self._translation.to(dtype=dtype),
            matrix_shape=self._matrix_shape,
            dtype=dtype,
            device=device,
        )

    def __repr__(self) -> str:
        return (
            f"HostDiagonalAffineTransformation("
            f"diagonal={self._diagonal}, translation={self._translation}, "
            f"matrix_shape={self._matrix_shape}, dtype={self._dtype}, device={self._target_device})"
        )

    def __neg__(self) -> "HostDiagonalAffineTransformation":
        negated_diagonal, negated_translation = negate_diagonal_affine_matrix(
            self._diagonal,
            self._translation,
            matrix_shape=self._matrix_shape,
            dtype=self._dtype,
            device=self._device,
        )
        return HostDiagonalAffineTransformation(
            diagonal=negated_diagonal,
            translation=negated_translation,
            matrix_shape=self._matrix_shape,
            dtype=self._dtype,
            device=self._target_device,
        )

    def invert(self) -> "HostDiagonalAffineTransformation":
        cpu_diagonal, cpu_translation = self.as_cpu_diagonal()
        inverted_cpu_diagonal, inverted_cpu_translation = invert_diagonal_affine_matrix(
            cpu_diagonal, cpu_translation, matrix_shape=self._matrix_shape
        )
        return HostDiagonalAffineTransformation(
            diagonal=inverted_cpu_diagonal,
            translation=inverted_cpu_translation,
            matrix_shape=self._matrix_shape,
            dtype=self._dtype,
            device=self._target_device,
        )


class IdentityAffineTransformation(HostDiagonalAffineTransformation):
    """Identity affine transformation"""

    def __init__(
        self,
        n_dims: int,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> None:
        super().__init__(
            diagonal=None,
            translation=None,
            matrix_shape=(n_dims + 1, n_dims + 1),
            dtype=dtype,
            device=device,
        )
        self._n_dims = n_dims

    def __repr__(self) -> str:
        return (
            f"IdentityAffineTransformation("
            f"n_dims={self._n_dims}, "
            f"dtype={self._dtype}, "
            f"device={self._target_device})"
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
        scales: Tensor with shape (batch_size, n_dims, ...)

    Returns: Tensor with shape (batch_size, n_dims, n_dims, ...)
    """
    matrix_dims = get_channel_dims(scales.ndim + 1, n_channel_dims=2)
    scales = move_channels_last(scales, n_channel_dims=1)
    return diag_embed(scales, dim1=matrix_dims[0], dim2=matrix_dims[1])


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


def _get_diagonal_matrix_shape_splitted(
    diagonal: Optional[Tensor], translation: Optional[Tensor], matrix_shape: Optional[Sequence[int]]
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    if matrix_shape is None:
        if diagonal is not None and translation is not None:
            channels_shape: Tuple[int, ...] = (
                get_channels_shape(translation.shape, n_channel_dims=1)[0] + 1,
                get_channels_shape(diagonal.shape, n_channel_dims=1)[0] + 1,
            )
        elif diagonal is not None:
            channels_shape = (
                get_channels_shape(diagonal.shape, n_channel_dims=1)[0] + 1,
                get_channels_shape(diagonal.shape, n_channel_dims=1)[0] + 1,
            )
        elif translation is not None:
            channels_shape = (
                get_channels_shape(translation.shape, n_channel_dims=1)[0] + 1,
                get_channels_shape(translation.shape, n_channel_dims=1)[0] + 1,
            )
        else:
            raise ValueError(
                "At least one of diagonal, translation, and matrix_shape must be given."
            )
    else:
        channels_shape = get_channels_shape(matrix_shape, n_channel_dims=2)
    if diagonal is not None:
        diagonal_length = get_channels_shape(diagonal.shape, n_channel_dims=1)[0]
        if diagonal_length + 1 != channels_shape[0] or diagonal_length + 1 != channels_shape[1]:
            raise ValueError("The diagonal does not match the matrix shape.")
    if (
        translation is not None
        and get_channels_shape(translation.shape, n_channel_dims=1)[0] + 1 != channels_shape[0]
    ):
        raise ValueError("The translation does not match the matrix shape.")
    batch_size, _, spatial_shape = broadcast_optional_shapes_in_parts_splitted(
        diagonal.shape if diagonal is not None else None,
        translation.shape if translation is not None else None,
        matrix_shape,
        n_channel_dims=(1, 1, 2),
        broadcast_channels=False,
    )
    assert batch_size is not None and spatial_shape is not None
    return batch_size, channels_shape, spatial_shape


def generate_diagonal_affine_matrix(
    diagonal: Optional[Tensor],
    translation: Optional[Tensor],
    matrix_shape: Optional[Sequence[int]],
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> Tensor:
    """Generator diagonal affine matrix from given diagonal and translation

    Args:
        diagonal: Tensor with shape (batch_size, n_dims, ...)
        translation: Tensor with shape (batch_size, n_dims, ...)

    Returns: Tensor with shape (batch_size, n_dims + 1, n_dims + 1, ...)
    """
    diagonal, translation, dtype, device = _handle_diagonal_input_dtype_and_device(
        diagonal, translation, dtype, device
    )
    batch_shape, channels_shape, spatial_shape = _get_diagonal_matrix_shape_splitted(
        diagonal, translation, matrix_shape
    )
    if diagonal is not None:
        matrix = generate_scale_matrix(diagonal)
    else:
        matrix = eye(
            channels_shape[0] - 1,
            channels_shape[1] - 1,
            dtype=dtype,
            device=device,
        )
    matrix = broadcast_to_in_parts(
        matrix,
        batch_shape=batch_shape,
        spatial_shape=spatial_shape,
        n_channel_dims=2,
    )
    if translation is not None:
        translation = broadcast_to_in_parts(
            translation,
            batch_shape=batch_shape,
            spatial_shape=spatial_shape,
        )
        matrix = embed_matrix(matrix, (channels_shape[0] - 1, channels_shape[1] - 1))
        matrix_channel_dims = get_channel_dims(matrix.ndim, n_channel_dims=2)
        matrix = cat(
            (matrix, translation.unsqueeze(matrix_channel_dims[1])),
            dim=matrix_channel_dims[1],
        )
    matrix = embed_matrix(matrix, channels_shape)
    return matrix


def transform_values_with_diagonal_affine_matrix(
    diagonal: Optional[Tensor],
    translation: Optional[Tensor],
    values: Tensor,
    n_channel_dims: int = 1,
    matrix_shape: Optional[Sequence[int]] = None,
) -> Tensor:
    """Transform values with affine matrix"""
    affine_batch_shape, affine_channels_shape, affine_spatial_shape = (
        _get_diagonal_matrix_shape_splitted(diagonal, translation, matrix_shape)
    )
    batch_shape, _, spatial_shape = broadcast_shapes_in_parts_splitted(
        values.shape,
        affine_batch_shape + affine_channels_shape + affine_spatial_shape,
        n_channel_dims=(n_channel_dims, 2),
        broadcast_channels=False,
    )
    n_input_dims = get_channels_shape(values.shape, n_channel_dims=n_channel_dims)[-1]
    if affine_channels_shape[1] - 1 != n_input_dims:
        raise ValueError("The diagonal matrix does not match the number of dimensions.")
    diagonal_length = min(affine_channels_shape[0] - 1, affine_channels_shape[1] - 1)
    if diagonal_length < n_input_dims:
        values = move_channels_first(
            move_channels_last(values, n_channel_dims)[..., :diagonal_length], n_channel_dims
        )
    if diagonal is not None:
        diagonal = broadcast_to_in_parts(
            diagonal,
            batch_shape=batch_shape,
            channels_shape=get_channels_shape(values.shape, n_channel_dims=n_channel_dims),
            spatial_shape=spatial_shape,
            n_channel_dims=1,
        )
        values = values * diagonal
    n_output_dims = affine_channels_shape[0] - 1
    if diagonal_length < n_output_dims:
        last_values_channel_dim = get_channel_dims(values.ndim, n_channel_dims=n_channel_dims)[-1]
        values = _pad_dimension_with_zeros(values, n_output_dims, last_values_channel_dim)
    if translation is not None:
        translation = broadcast_to_in_parts(
            translation,
            batch_shape=batch_shape,
            channels_shape=get_channels_shape(values.shape, n_channel_dims=n_channel_dims),
            spatial_shape=spatial_shape,
            n_channel_dims=1,
        )
        values = values + translation
    return values


def invert_diagonal_affine_matrix(
    diagonal: Optional[Tensor],
    translation: Optional[Tensor],
    matrix_shape: Optional[Sequence[int]] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Transform values with affine matrix"""
    _affine_batch_shape, affine_channels_shape, _affine_spatial_shape = (
        _get_diagonal_matrix_shape_splitted(diagonal, translation, matrix_shape)
    )
    if affine_channels_shape[0] != affine_channels_shape[1]:
        raise ValueError("The diagonal matrix must be square.")
    if diagonal is not None:
        diagonal = 1 / diagonal
    if translation is not None:
        translation = -translation
        if diagonal is not None:
            diagonal, translation = broadcast_tensors_in_parts(
                diagonal, translation, n_channel_dims=1
            )
            translation = translation * diagonal
    return diagonal, translation


def negate_diagonal_affine_matrix(
    diagonal: Optional[Tensor],
    translation: Optional[Tensor],
    matrix_shape: Optional[Sequence[int]] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Transform values with affine matrix"""
    diagonal, translation, dtype, device = _handle_diagonal_input_dtype_and_device(
        diagonal, translation, dtype, device
    )
    _affine_batch_shape, affine_channels_shape, _affine_spatial_shape = (
        _get_diagonal_matrix_shape_splitted(diagonal, translation, matrix_shape)
    )
    if translation is not None:
        translation = -translation
    if diagonal is not None:
        diagonal = -diagonal
    else:
        diagonal_length = min(affine_channels_shape[0] - 1, affine_channels_shape[1] - 1)
        diagonal = -ones(diagonal_length, dtype=dtype, device=device)
    return diagonal, translation


def is_identity_diagonal_affine_matrix(
    diagonal: Optional[Tensor],
    translation: Optional[Tensor],
) -> bool:
    """Return whether a diagonal matrix is an identity"""
    if diagonal is not None:
        if not allclose(
            diagonal,
            ones(1, dtype=diagonal.dtype, device=diagonal.device),
            atol=IDENTITY_MATRIX_TOLERANCE,
        ):
            return False
    if translation is not None:
        if not allclose(
            translation,
            zeros(1, dtype=translation.dtype, device=translation.device),
            atol=IDENTITY_MATRIX_TOLERANCE,
        ):
            return False
    return True


def is_zero_diagonal_affine_matrix(
    diagonal: Optional[Tensor],
    translation: Optional[Tensor],
) -> bool:
    """Return whether a diagonal matrix is a zero matrix"""
    if diagonal is not None:
        if not allclose(
            diagonal,
            zeros(1, dtype=diagonal.dtype, device=diagonal.device),
            atol=ZERO_MATRIX_TOLERANCE,
        ):
            return False
    if translation is not None:
        if not allclose(
            translation,
            zeros(1, dtype=translation.dtype, device=translation.device),
            atol=ZERO_MATRIX_TOLERANCE,
        ):
            return False
    return True


def compose_diagonal_affine_matrices(
    diagonal_1: Optional[Tensor],
    translation_1: Optional[Tensor],
    matrix_shape_1: Optional[Sequence[int]],
    diagonal_2: Optional[Tensor],
    translation_2: Optional[Tensor],
    matrix_shape_2: Optional[Sequence[int]],
) -> Tuple[Optional[Tensor], Optional[Tensor], Sequence[int]]:
    """Compose two diagonal affine matrices"""
    batch_shape_1, channels_shape_1, spatial_shape_1 = _get_diagonal_matrix_shape_splitted(
        diagonal_1, translation_1, matrix_shape_1
    )
    batch_shape_2, channels_shape_2, spatial_shape_2 = _get_diagonal_matrix_shape_splitted(
        diagonal_2, translation_2, matrix_shape_2
    )
    diagonal_length_1 = min(channels_shape_1[0] - 1, channels_shape_1[1] - 1)
    diagonal_length_2 = min(channels_shape_2[0] - 1, channels_shape_2[1] - 1)
    shared_diagonal_length = min(diagonal_length_1, diagonal_length_2)
    batch_shape = broadcast_shapes(batch_shape_1, batch_shape_2)
    spatial_shape = broadcast_shapes(spatial_shape_1, spatial_shape_2)
    if channels_shape_1[1] != channels_shape_2[0]:
        raise ValueError("The matrices are not compatible.")
    channels_shape = (channels_shape_1[0], channels_shape_2[1])
    diagonal_length = min(channels_shape[0] - 1, channels_shape[1] - 1)

    if diagonal_1 is not None and shared_diagonal_length < diagonal_length_1:
        channel_dim = get_channel_dims(diagonal_1.ndim, n_channel_dims=1)[0]
        diagonal_1 = diagonal_1.narrow(channel_dim, 0, diagonal_length)
    if diagonal_2 is not None and shared_diagonal_length < diagonal_length_2:
        channel_dim = get_channel_dims(diagonal_2.ndim, n_channel_dims=1)[0]
        diagonal_2 = diagonal_2.narrow(channel_dim, 0, diagonal_length)

    if diagonal_1 is not None and diagonal_2 is not None:
        diagonal_1, diagonal_2 = broadcast_tensors_in_parts(
            diagonal_1, diagonal_2, n_channel_dims=1
        )
        diagonal: Optional[Tensor] = diagonal_1 * diagonal_2
    elif diagonal_1 is not None:
        diagonal = diagonal_1
    elif diagonal_2 is not None:
        diagonal = diagonal_2
    else:
        diagonal = None
    if diagonal is not None:
        channel_dim = get_channel_dims(diagonal.ndim, n_channel_dims=1)[0]
        diagonal = _pad_dimension_with_zeros(diagonal, diagonal_length, channel_dim)

    if translation_2 is not None:
        channel_dim = get_channel_dims(translation_2.ndim, n_channel_dims=1)[0]
        translation_2 = translation_2.narrow(channel_dim, 0, diagonal_length_1)
        if diagonal_1 is not None:
            diagonal_1, translation_2 = broadcast_tensors_in_parts(
                diagonal_1, translation_2, n_channel_dims=1
            )
            translation_2 = translation_2 * diagonal_1
        channel_dim = get_channel_dims(translation_2.ndim, n_channel_dims=1)[0]
        translation_2 = _pad_dimension_with_zeros(translation_2, channels_shape[0] - 1, channel_dim)

    if translation_1 is not None and translation_2 is not None:
        translation_1, translation_2 = broadcast_tensors_in_parts(
            translation_1, translation_2, n_channel_dims=1
        )
        translation: Optional[Tensor] = translation_2 + translation_1
    elif translation_1 is not None:
        translation = translation_1
    elif translation_2 is not None:
        translation = translation_2
    else:
        translation = None

    return diagonal, translation, batch_shape + channels_shape + spatial_shape


def add_diagonal_affine_matrices(
    diagonal_1: Optional[Tensor],
    translation_1: Optional[Tensor],
    matrix_shape_1: Optional[Sequence[int]],
    diagonal_2: Optional[Tensor],
    translation_2: Optional[Tensor],
    matrix_shape_2: Optional[Sequence[int]],
) -> Tuple[Optional[Tensor], Optional[Tensor], Sequence[int]]:
    """Add two diagonal affine matrices"""
    batch_shape_1, channels_shape_1, spatial_shape_1 = _get_diagonal_matrix_shape_splitted(
        diagonal_1, translation_1, matrix_shape_1
    )
    batch_shape_2, channels_shape_2, spatial_shape_2 = _get_diagonal_matrix_shape_splitted(
        diagonal_2, translation_2, matrix_shape_2
    )
    batch_shape = broadcast_shapes(batch_shape_1, batch_shape_2)
    spatial_shape = broadcast_shapes(spatial_shape_1, spatial_shape_2)
    if channels_shape_1 != channels_shape_2:
        raise ValueError("The matrices are not compatible.")

    if diagonal_1 is not None and diagonal_2 is not None:
        diagonal_1, diagonal_2 = broadcast_tensors_in_parts(
            diagonal_1, diagonal_2, n_channel_dims=1
        )
        diagonal: Optional[Tensor] = diagonal_1 + diagonal_2
    elif diagonal_1 is not None:
        diagonal = diagonal_1
    elif diagonal_2 is not None:
        diagonal = diagonal_2
    else:
        diagonal = None

    if translation_1 is not None and translation_2 is not None:
        translation_1, translation_2 = broadcast_tensors_in_parts(
            translation_1, translation_2, n_channel_dims=1
        )
        translation: Optional[Tensor] = translation_2 + translation_1
    elif translation_1 is not None:
        translation = translation_1
    elif translation_2 is not None:
        translation = translation_2
    else:
        translation = None

    return diagonal, translation, batch_shape + channels_shape_1 + spatial_shape


def _pad_dimension_with_zeros(item: Tensor, new_size: int, dim: int) -> Tensor:
    current_size = item.size(dim)
    return cat(
        (
            item,
            zeros(1, device=item.device, dtype=item.dtype).expand(
                *item.shape[:dim],
                new_size - current_size,
                *item.shape[dim + 1 :],
            ),
        ),
        dim=dim,
    )


def _handle_diagonal_input_dtype_and_device(
    diagonal: Optional[Tensor],
    translation: Optional[Tensor],
    dtype: Optional[torch_dtype],
    device: Optional[torch_device],
) -> Tuple[Optional[Tensor], Optional[Tensor], torch_dtype, torch_device]:
    if diagonal is not None:
        if device is not None and device != diagonal.device:
            raise ValueError(
                "Device mismatch, note that the device is needed only if no diagonal or "
                "translation is given."
            )
        if dtype is not None and dtype != diagonal.dtype:
            raise ValueError(
                "Dtype mismatch, note that the dtype is needed only if no diagonal or "
                "translation is given."
            )
        dtype = diagonal.dtype
        device = diagonal.device
    elif translation is not None:
        if device is not None and device != translation.device:
            raise ValueError(
                "Device mismatch, note that the device is needed only if no diagonal or "
                "translation is given."
            )
        if dtype is not None and dtype != translation.dtype:
            raise ValueError(
                "Dtype mismatch, note that the dtype is needed only if no diagonal or "
                "translation is given."
            )
        dtype = translation.dtype
        device = translation.device
    dtype = get_default_dtype() if dtype is None else dtype
    device = torch_device("cpu") if device is None else device
    return diagonal, translation, dtype, device
