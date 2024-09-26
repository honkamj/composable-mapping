"""Interfaces for composable mapping"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Sequence, Tuple, TypeVar, Union, overload

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype

ITensorLikeT = TypeVar("ITensorLikeT", bound="ITensorLike")


class ITensorLike(ABC):
    """Interface for classes having tensor like properties

    Usually contains wrapped tensors with device, dtype, and detachment
    related functionalities corresponding directly to the wrapped tensors.
    """

    @property
    @abstractmethod
    def dtype(
        self,
    ) -> torch_dtype:
        """Return the dtype of the underlying tensor(s)"""

    @property
    @abstractmethod
    def device(
        self,
    ) -> torch_device:
        """Return the device of the underlying tensor(s)"""

    @abstractmethod
    def cast(
        self: ITensorLikeT,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> ITensorLikeT:
        """Cast underlying tensors to given dtype and device

        We will not use method name "to" as it would create conflict with
        torch.nn.Module.to method which does casting in-place.
        """

    @abstractmethod
    def detach(self: ITensorLikeT) -> ITensorLikeT:
        """Detach the wrapped tensors from computational graph"""


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
        """Returns the transformation matrix on cpu, if available"""

    @property
    @abstractmethod
    def n_input_dims(self) -> int:
        """Number of dimensions the transformation expects"""

    @property
    @abstractmethod
    def n_output_dims(self) -> int:
        """Number of dimensions the transformation outputs"""

    @abstractmethod
    def is_zero(self, n_input_dims: int, n_output_dims: int) -> Optional[bool]:
        """Return whether the transformation is zero

        Returns None if the check cannot be done on CPU

        Args:
            n_input_dims: Number of dimensions the transformation is applied on
            n_output_dims: Number of dimensions the transformation outputs
        """


class IMaskedTensor(ITensorLike):
    """Wrapper for masked tensor"""

    @overload
    def generate(
        self,
        generate_missing_mask: Literal[True] = True,
        cast_mask: bool = ...,
    ) -> Tuple[Tensor, Tensor]: ...

    @overload
    def generate(
        self,
        generate_missing_mask: bool,
        cast_mask: bool = ...,
    ) -> Tuple[Tensor, Optional[Tensor]]: ...

    @abstractmethod
    def generate(
        self,
        generate_missing_mask: bool = True,
        cast_mask: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Generate values and mask contained by the wrapper

        Args:
            generate_mask: Generate mask of ones if the object does not have a mask
            cast_mask: Mask is stored as a boolean tensor, cast it to dtype of values if True
        """

    @abstractmethod
    def generate_values(
        self,
    ) -> Tensor:
        """Generate values contained by the wrapper"""

    @overload
    def generate_mask(
        self,
        generate_missing_mask: Literal[True] = ...,
        cast_mask: bool = ...,
    ) -> Tensor: ...

    @overload
    def generate_mask(
        self,
        generate_missing_mask: Union[bool, Literal[False]],
        cast_mask: bool = ...,
    ) -> Optional[Tensor]: ...

    @abstractmethod
    def generate_mask(
        self,
        generate_missing_mask: bool = True,
        cast_mask: bool = False,
    ) -> Optional[Tensor]:
        """Generate mask contained by the wrapper

        Args:
            generate_mask: Generate mask of ones if the object does not have a mask
            cast_mask: Mask is stored as a boolean tensor, cast it to dtype of values if True
        """

    @abstractmethod
    def has_mask(self) -> bool:
        """Returns whether the tensor has a mask"""

    @abstractmethod
    def apply_affine(self, affine_transformation: IAffineTransformation) -> "IMaskedTensor":
        """Apply affine mapping to the first channel dimension of the tensor"""

    @abstractmethod
    def add_grid(
        self,
        affine_transformation: IAffineTransformation,
    ) -> "IMaskedTensor":
        """Add affine transformation to the existing affine transformatoin on
        the voxel grid of the tensor"""

    @property
    @abstractmethod
    def channels_shape(self) -> Sequence[int]:
        """Return shape of the channel dimensions"""

    @property
    @abstractmethod
    def n_channel_dims(self) -> int:
        """Return number of channel dimensions"""

    @property
    @abstractmethod
    def spatial_shape(self) -> Sequence[int]:
        """Return shape of the spatial dimensions"""

    @property
    @abstractmethod
    def shape(self) -> Sequence[int]:
        """Shape of the values"""

    @abstractmethod
    def clear_mask(self) -> "IMaskedTensor":
        """Return version of the tensor with mask cleared"""

    @abstractmethod
    def as_slice(
        self, target_shape: Sequence[int]
    ) -> Optional[Tuple[Union["ellipsis", slice], ...]]:
        """If the masked tensor is a grid, provide a slice that can be used to
        extract a subgrid on a volume with given target shape

        Assumes that the target volume is in voxel coordinates.

        Should be done on CPU, if impossible, returns None"""

    @abstractmethod
    def reduce(self) -> "IMaskedTensor":
        """Return the masked tensor with the values explicitly stored"""

    @abstractmethod
    def displace(self, displacement: Tensor) -> "IMaskedTensor":
        """Return a masked tensor with the values displaced"""

    @abstractmethod
    def modify_values(self, values: Tensor) -> "IMaskedTensor":
        """Return a masked tensor with the values modified

        Assumes that the number of channel dimensions is the same as the
        original tensor
        """

    @abstractmethod
    def modify_mask(self, mask: Optional[Tensor]) -> "IMaskedTensor":
        """Return a masked tensor with the mask modified

        Setting mask to None is equivalent to clearing the mask
        """

    @abstractmethod
    def modify(self, values: Tensor, mask: Optional[Tensor]) -> "IMaskedTensor":
        """Return a masked tensor with values and mask modified

        Assumes that the number of channel dimensions is the same as the
        original tensor
        """


class IComposableMapping(ITensorLike):
    """Composable mapping"""

    @abstractmethod
    def compose(self, mapping: "IComposableMapping") -> "IComposableMapping":
        """Compose with another mapping"""

    @abstractmethod
    def __call__(self, masked_coordinates: IMaskedTensor) -> IMaskedTensor:
        """Evaluate the mapping at coordinates"""

    @abstractmethod
    def invert(self, **inversion_parameters) -> "IComposableMapping":
        """Invert the mapping

        Args:
            inversion_parameters: Possible inversion parameters
        """


class IInterpolator(ABC):
    """Interpolates values on regular grid in voxel coordinates"""

    @abstractmethod
    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        """Interpolate

        Args:
            volume: Volume to be interpolated with shape
                (batch_size, *channel_dims, dim_1, ..., dim_{n_dims}). Dimension
                order is the same as the coordinate order of the coordinates
            coordinates: Interpolation coordinates with shape
                (batch_size, n_dims, *target_shape)

        Returns:
            Interpolated volume with shape (batch_size, *channel_dims, *target_shape)
        """
