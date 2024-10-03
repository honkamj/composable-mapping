"""Affine transformation implementations"""

from abc import abstractmethod
from typing import Mapping, Optional, Sequence, Tuple, Union, overload

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import zeros

from composable_mapping.mappable_tensor.diagonal_matrix import (
    DiagonalAffineMatrixDefinition,
)
from composable_mapping.tensor_like import BaseTensorLikeWrapper, ITensorLike
from composable_mapping.util import (
    broadcast_shapes_in_parts,
    broadcast_to_in_parts,
    get_channel_dims,
    split_shape,
)

from .diagonal_matrix import (
    DiagonalAffineMatrixDefinition,
    add_diagonal_affine_matrices,
    compose_diagonal_affine_matrices,
    invert_diagonal_affine_matrix,
    is_identity_diagonal_affine_matrix,
    is_zero_diagonal_affine_matrix,
    negate_diagonal_affine_matrix,
    transform_values_with_diagonal_affine_matrix,
)
from .matrix import (
    add_affine_matrices,
    compose_affine_matrices,
    invert_matrix,
    is_identity_matrix,
    is_zero_matrix,
    negate_affine_matrix,
    transform_values_with_affine_matrix,
)

Number = Union[int, float]


class IAffineTransformation(ITensorLike):
    """Affine transformation"""

    @abstractmethod
    def __matmul__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __add__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __sub__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __neg__(self) -> "IAffineTransformation": ...

    @abstractmethod
    def invert(self) -> "IAffineTransformation":
        """Invert the transformation"""

    @abstractmethod
    def as_matrix(
        self,
    ) -> Tensor:
        """Return the mapping as matrix"""

    @abstractmethod
    def as_host_matrix(self) -> Optional[Tensor]:
        """Return detach transformation matrix detached on host (cpu), if available"""

    @abstractmethod
    def as_diagonal(
        self,
    ) -> Optional[DiagonalAffineMatrixDefinition]:
        """Return the mapping as diagonal matrix, if possible"""

    @abstractmethod
    def as_host_diagonal(
        self,
    ) -> Optional[DiagonalAffineMatrixDefinition]:
        """Return the mapping as diagonal matrix detached on host (cpu), if available"""

    @abstractmethod
    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        """Evaluate the mapping with values"""

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


class IHostAffineTransformation(IAffineTransformation):
    """Host affine transformation"""

    @overload
    def __matmul__(
        self, affine_transformation: "IHostAffineTransformation"
    ) -> "IHostAffineTransformation": ...

    @overload
    def __matmul__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __matmul__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @overload
    def __add__(
        self, affine_transformation: "IHostAffineTransformation"
    ) -> "IHostAffineTransformation": ...

    @overload
    def __add__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __add__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @overload
    def __sub__(
        self, affine_transformation: "IHostAffineTransformation"
    ) -> "IHostAffineTransformation": ...

    @overload
    def __sub__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def __sub__(
        self, affine_transformation: "IAffineTransformation"
    ) -> "IAffineTransformation": ...

    @abstractmethod
    def invert(self) -> "IHostAffineTransformation":
        """Invert the transformation"""

    @abstractmethod
    def as_host_matrix(self) -> Tensor:
        """Return detach transformation matrix detached on host (cpu)"""


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

    @overload
    def __matmul__(
        self, affine_transformation: IHostAffineTransformation
    ) -> IHostAffineTransformation: ...

    @overload
    def __matmul__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation: ...

    def __matmul__(self, affine_transformation: IAffineTransformation) -> "IAffineTransformation":
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        self_host_matrix = self.as_host_matrix()
        target_host_matrix = affine_transformation.as_host_matrix()
        if self_host_matrix is not None and target_host_matrix is not None:
            return HostAffineTransformation(
                transformation_matrix_on_host=compose_affine_matrices(
                    self_host_matrix, target_host_matrix
                ),
                device=self.device,
            )
        return AffineTransformation(
            compose_affine_matrices(
                self.as_matrix(),
                affine_transformation.as_matrix(),
            )
        )

    @overload
    def __add__(
        self, affine_transformation: IHostAffineTransformation
    ) -> IHostAffineTransformation: ...

    @overload
    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation: ...

    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        self_host_matrix = self.as_host_matrix()
        target_host_matrix = affine_transformation.as_host_matrix()
        if self_host_matrix is not None and target_host_matrix is not None:
            return HostAffineTransformation(
                transformation_matrix_on_host=add_affine_matrices(
                    self_host_matrix, target_host_matrix
                ),
                device=self.device,
            )
        return AffineTransformation(
            add_affine_matrices(self.as_matrix(), affine_transformation.as_matrix())
        )

    @overload
    def __sub__(
        self, affine_transformation: IHostAffineTransformation
    ) -> IHostAffineTransformation: ...

    @overload
    def __sub__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation: ...

    def __sub__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        return self.__add__(-affine_transformation)

    def is_zero(self) -> Optional[bool]:
        host_matrix = self.as_host_matrix()
        if host_matrix is not None:
            return is_zero_matrix(host_matrix)
        return None

    def as_diagonal(self) -> Optional[DiagonalAffineMatrixDefinition]:
        return None

    def as_host_diagonal(self) -> Optional[DiagonalAffineMatrixDefinition]:
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

    def as_host_matrix(self) -> Optional[Tensor]:
        return None

    def __neg__(self) -> IAffineTransformation:
        return AffineTransformation(negate_affine_matrix(self.as_matrix()))

    def invert(self) -> IAffineTransformation:
        return AffineTransformation(invert_matrix(self.as_matrix()))

    def __repr__(self) -> str:
        return f"AffineTransformation(transformation_matrix={self._transformation_matrix})"


class HostAffineTransformation(AffineTransformation, IHostAffineTransformation):
    """Affine tranformation for which matrix operations are done
    on host, and the matrix on target device is created only when needed

    Allows to do control flow decisions on host without having to do CPU-GPU
    synchronization.

    Arguments:
        transformation_matrix_on_host: Transformation matrix on cpu
        device: Device to use for the transformation matrix produced by as_matrix method
    """

    def __init__(
        self,
        transformation_matrix_on_host: Tensor,
        device: Optional[torch_device] = None,
    ) -> None:
        if transformation_matrix_on_host.device != torch_device("cpu"):
            raise ValueError("Please give the matrix on CPU")
        if transformation_matrix_on_host.requires_grad:
            raise ValueError("The implementation assumes a detached transformation matrix.")
        super().__init__(transformation_matrix=transformation_matrix_on_host)
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
        matrix = (
            super()
            .as_matrix()
            .to(device=self._device, non_blocking=self._device != torch_device("cpu"))
        )
        return matrix

    def as_host_matrix(self) -> Tensor:
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
        non_blocking: bool = False,
    ) -> "HostAffineTransformation":
        return HostAffineTransformation(
            transformation_matrix_on_host=self._transformation_matrix.to(
                dtype=self._transformation_matrix.dtype if dtype is None else dtype,
                non_blocking=non_blocking,
            ),
            device=self._device if device is None else device,
        )

    def __repr__(self) -> str:
        return (
            f"HostAffineTransformation("
            f"transformation_matrix_on_host={self._transformation_matrix}, "
            f"device={self._device})"
        )

    def __neg__(self) -> "HostAffineTransformation":
        return HostAffineTransformation(
            transformation_matrix_on_host=negate_affine_matrix(self.as_host_matrix()),
            device=self.device,
        )

    def invert(self) -> "HostAffineTransformation":
        return HostAffineTransformation(
            transformation_matrix_on_host=invert_matrix(self.as_host_matrix()),
            device=self.device,
        )


class BaseDiagonalAffineTransformation(BaseAffineTransformation):
    """Base diagonal affine transformation"""

    @overload
    def __matmul__(
        self, affine_transformation: IHostAffineTransformation
    ) -> IHostAffineTransformation: ...

    @overload
    def __matmul__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation: ...

    def __matmul__(self, affine_transformation: IAffineTransformation) -> "IAffineTransformation":
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        self_host_diagonal = self.as_host_diagonal()
        target_host_diagonal = affine_transformation.as_host_diagonal()
        if self_host_diagonal is not None and target_host_diagonal is not None:
            return HostDiagonalAffineTransformation.from_definition(
                compose_diagonal_affine_matrices(
                    self_host_diagonal,
                    target_host_diagonal,
                ),
                device=self.device,
            )

        self_diagonal = self.as_diagonal()
        target_diagonal = affine_transformation.as_diagonal()
        if self_diagonal is not None and target_diagonal is not None:
            return DiagonalAffineTransformation.from_definition(
                compose_diagonal_affine_matrices(
                    self_diagonal,
                    target_diagonal,
                )
            )
        return super().__matmul__(affine_transformation)

    @overload
    def __add__(
        self, affine_transformation: IHostAffineTransformation
    ) -> IHostAffineTransformation: ...

    @overload
    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation: ...

    def __add__(self, affine_transformation: IAffineTransformation) -> IAffineTransformation:
        if not isinstance(affine_transformation, IAffineTransformation):
            return NotImplemented
        if isinstance(affine_transformation, BaseDiagonalAffineTransformation):
            self_host_diagonal = self.as_host_diagonal()
            target_host_diagonal = affine_transformation.as_host_diagonal()
            if self_host_diagonal is not None and target_host_diagonal is not None:
                return HostDiagonalAffineTransformation.from_definition(
                    add_diagonal_affine_matrices(self_host_diagonal, target_host_diagonal),
                    device=self.device,
                )
            self_diagonal = self.as_diagonal()
            target_diagonal = affine_transformation.as_diagonal()
            if self_diagonal is not None and target_diagonal is not None:
                return DiagonalAffineTransformation.from_definition(
                    add_diagonal_affine_matrices(self_diagonal, target_diagonal)
                )
        return super().__add__(affine_transformation)

    @abstractmethod
    def as_diagonal(self) -> DiagonalAffineMatrixDefinition:
        """Return the diagonal and translation tensors"""

    @abstractmethod
    def as_host_diagonal(self) -> Optional[DiagonalAffineMatrixDefinition]:
        """Return the diagonal and translation tensors detached on cpu, if available"""


class DiagonalAffineTransformation(BaseTensorLikeWrapper, BaseDiagonalAffineTransformation):
    """Represents diagonal affine transformation

    Arguments:
        diagonal: Tensor with shape ([batch_size, ]diagonal_length, ...),
            if None, corresponds to all ones
        translation: Tensor with shape ([batch_size, ]n_target_dims, ...),
            if None, corresponds to all zeros
        matrix_shape: Shape of the target affine transformation matrix
            (n_target_dims + 1, n_source_dims + 1)
    """

    def __init__(
        self,
        diagonal: Optional[Union[Tensor, Number]] = None,
        translation: Optional[Union[Tensor, Number]] = None,
        matrix_shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> None:
        self._matrix_definition = DiagonalAffineMatrixDefinition(
            diagonal=diagonal,
            translation=translation,
            matrix_shape=matrix_shape,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_definition(
        cls, matrix_definition: DiagonalAffineMatrixDefinition
    ) -> "DiagonalAffineTransformation":
        """Create diagonal affine transformation from definition"""
        instance = cls.__new__(cls)
        instance._matrix_definition = matrix_definition
        return instance

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {"matrix_definition": self._matrix_definition}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "DiagonalAffineTransformation":
        if not isinstance(children["matrix_definition"], DiagonalAffineMatrixDefinition):
            raise ValueError("Invalid children for DiagonalAffineTransformation")
        return DiagonalAffineTransformation.from_definition(children["matrix_definition"])

    @property
    def matrix_shape(self) -> Sequence[int]:
        return self._matrix_definition.matrix_shape

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        return transform_values_with_diagonal_affine_matrix(
            self.as_diagonal(),
            values,
            n_channel_dims=n_channel_dims,
        )

    def as_matrix(
        self,
    ) -> Tensor:
        return self._matrix_definition.as_matrix()

    def as_host_matrix(self) -> Optional[Tensor]:
        return None

    def as_diagonal(
        self,
    ) -> DiagonalAffineMatrixDefinition:
        return self._matrix_definition

    def as_host_diagonal(self) -> Optional[DiagonalAffineMatrixDefinition]:
        return None

    def __neg__(self) -> IAffineTransformation:
        return DiagonalAffineTransformation.from_definition(
            negate_diagonal_affine_matrix(self._matrix_definition)
        )

    def invert(self) -> IAffineTransformation:
        return DiagonalAffineTransformation.from_definition(
            invert_diagonal_affine_matrix(self._matrix_definition)
        )

    def __repr__(self) -> str:
        return f"DiagonalAffineTransformation(definition={self._matrix_definition})"

    def is_zero(self) -> Optional[bool]:
        return None


class HostDiagonalAffineTransformation(DiagonalAffineTransformation, IHostAffineTransformation):
    """Diagonal affine tranformation for which matrix operations are done
    on host, and the matrix on target device is created only when needed

    Allows to do control flow decisions on host without having to do CPU-GPU
    synchronization.
    """

    def __init__(
        self,
        diagonal: Optional[Union[Tensor, Number]] = None,
        translation: Optional[Union[Tensor, Number]] = None,
        matrix_shape: Optional[Sequence[int]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> None:
        super().__init__(
            diagonal=diagonal,
            translation=translation,
            matrix_shape=matrix_shape,
            dtype=dtype,
            device=torch_device("cpu"),
        )
        self._target_device = torch_device("cpu") if device is None else device

    @classmethod
    def from_definition(
        cls,
        matrix_definition: DiagonalAffineMatrixDefinition,
        device: Optional[torch_device] = None,
    ) -> "HostDiagonalAffineTransformation":
        """Create diagonal affine transformation from definition"""
        instance = cls.__new__(cls)
        instance._matrix_definition = matrix_definition
        instance._target_device = torch_device("cpu") if device is None else device
        return instance

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "HostDiagonalAffineTransformation":
        if not isinstance(children["matrix_definition"], DiagonalAffineMatrixDefinition):
            raise ValueError("Invalid children for HostDiagonalAffineTransformation")
        return HostDiagonalAffineTransformation.from_definition(
            children["matrix_definition"], self.device
        )

    def __call__(self, values: Tensor, n_channel_dims: int = 1) -> Tensor:
        if is_identity_diagonal_affine_matrix(self._matrix_definition):
            batch_shape, channels_shape, spatial_shape = split_shape(
                self.get_output_shape(values.shape, n_channel_dims), n_channel_dims=n_channel_dims
            )
            return broadcast_to_in_parts(
                values,
                batch_shape=batch_shape,
                channels_shape=channels_shape,
                spatial_shape=spatial_shape,
                n_channel_dims=n_channel_dims,
            )
        if self.is_zero():
            return zeros(
                self.get_output_shape(values.shape, n_channel_dims),
                device=self._target_device,
                dtype=values.dtype,
            )
        return super().__call__(values, n_channel_dims)

    def as_matrix(
        self,
    ) -> Tensor:
        matrix = (
            super()
            .as_matrix()
            .to(device=self.device, non_blocking=self.device != torch_device("cpu"))
        )
        return matrix

    def as_host_matrix(self) -> Tensor:
        return super().as_matrix()

    def as_diagonal(self) -> DiagonalAffineMatrixDefinition:
        matrix_definition = super().as_diagonal()
        if self._target_device != torch_device("cpu"):
            matrix_definition = matrix_definition.cast(
                device=self._target_device, non_blocking=self.device != torch_device("cpu")
            )
        return matrix_definition

    def as_host_diagonal(self) -> DiagonalAffineMatrixDefinition:
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
        non_blocking: bool = False,
    ) -> "HostDiagonalAffineTransformation":
        dtype = self.dtype if dtype is None else dtype
        device = self.device if device is None else device
        return HostDiagonalAffineTransformation.from_definition(
            self._matrix_definition.cast(dtype=dtype, non_blocking=non_blocking), device=device
        )

    def __repr__(self) -> str:
        return f"HostDiagonalAffineTransformation(definition={self._matrix_definition})"

    def __neg__(self) -> "HostDiagonalAffineTransformation":
        return HostDiagonalAffineTransformation.from_definition(
            negate_diagonal_affine_matrix(self._matrix_definition), device=self.device
        )

    def invert(self) -> "HostDiagonalAffineTransformation":
        return HostDiagonalAffineTransformation.from_definition(
            invert_diagonal_affine_matrix(self._matrix_definition), device=self.device
        )

    def is_zero(self) -> bool:
        host_diagonal_matrix = self.as_host_diagonal()
        return is_zero_diagonal_affine_matrix(host_diagonal_matrix)


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
            f"dtype={self.dtype}, "
            f"device={self.device})"
        )
