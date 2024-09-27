"""Voxel coordinate system"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union, overload

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import eye, get_default_dtype, ones_like, tensor
from torch.nn import Module

from .affine_transformation import (
    CPUAffineTransformation,
    compose_affine_matrices,
    embed_matrix,
    generate_scale_matrix,
    generate_translation_matrix,
)
from .composable_affine import ComposableAffine
from .interface import IComposableMapping
from .mappable_tensor import MappableTensor, VoxelGrid
from .tensor_like import ITensorLike
from .util import ceildiv


class IVoxelCoordinateSystemContainer(ABC):
    """Class holding a unique voxel coordinate system"""

    @property
    @abstractmethod
    def coordinate_system(
        self,
    ) -> "VoxelCoordinateSystem":
        """Get voxel coordinate system of the container"""


Number = Union[float, int]


class IReformattingShapeOption(ABC):
    """Option for determining target shape based on the original shape and downsampling factor"""

    @abstractmethod
    def to_target_size(self, original_size: int, downsampling_factor: float) -> int:
        """Return the target size given the original size and downsampling factor"""


class RetainShapeOption(IReformattingShapeOption):
    """Option for retaining the original shape"""

    def to_target_size(self, original_size: int, downsampling_factor: float) -> int:
        return original_size


class FitToFOVOption(IReformattingShapeOption):
    """Option for reformatting target shape to fit the field of view size"""

    DIVISION_FUNCTIONS = {
        "round": lambda x, y: int(round(x / y)),
        "floor": lambda x, y: int(x // y),
        "ceil": lambda x, y: int(ceildiv(x, y)),
    }

    def __init__(self, fitting_method: str = "round", fov_convention: str = "full_voxels") -> None:
        if fitting_method not in ("round", "floor", "ceil"):
            raise ValueError(f"Unknown fitting method {fitting_method}")
        if fov_convention not in ("full_voxels", "voxel_centers"):
            raise ValueError(f"Unknown fov convention {fov_convention}")
        self._division_function = self.DIVISION_FUNCTIONS[fitting_method]
        self._fov_convention = fov_convention

    def to_target_size(self, original_size: int, downsampling_factor: float) -> int:
        if self._fov_convention == "full_voxels":
            return self._division_function(original_size, downsampling_factor)
        elif self._fov_convention == "voxel_centers":
            return self._division_function(original_size - 1, downsampling_factor) + 1
        raise ValueError(f"Unknown fov convention {self._fov_convention}")


class IReformattingReferenceOption(ABC):
    """Option for defining reference point which will be aligned between the
    original and the reformatted coordinates"""

    @abstractmethod
    def get_voxel_coordinate(self, size: int) -> float:
        """Get the voxel coordinate corresponding to the position with the given
        dimension size"""


class ReferenceOption(IReformattingReferenceOption):
    """Option for defining reformatting reference point"""

    def __init__(
        self,
        position: str = "left",
        fov_convention: str = "full_voxels",
    ) -> None:
        if position not in ("left", "right", "center"):
            raise ValueError(f"Unknown position {position}")
        if fov_convention not in ("full_voxels", "voxel_centers"):
            raise ValueError(f"Unknown fov convention {fov_convention}")
        self._position = position
        self._fov_convention = fov_convention

    def get_voxel_coordinate(self, size: int) -> float:
        if self._position == "center":
            return (size - 1) / 2
        if self._fov_convention == "full_voxels":
            if self._position == "left":
                return -0.5
            if self._position == "right":
                return size - 0.5
        elif self._fov_convention == "voxel_centers":
            if self._position == "left":
                return 0
            if self._position == "right":
                return size - 1
        raise ValueError(f"Unknown position {self._position}")


class VoxelCoordinateSystem(Module, IVoxelCoordinateSystemContainer, ITensorLike):
    """Represents coordinate system between voxel and world coordinates

    Arguments:
        from_voxel_coordinates: Affine transformation from voxel to world coordinates,
            the tensor should be on CPU
        shape: Shape of the grid in the voxel coordinates
        device: Device of the coordinate system
    """

    def __init__(
        self,
        shape: Sequence[int],
        to_voxel_coordinates: Optional[Tensor] = None,
        from_voxel_coordinates: Optional[Tensor] = None,
        device: Optional[torch_device] = None,
    ):
        super().__init__()
        if from_voxel_coordinates is None and to_voxel_coordinates is None:
            from_voxel_coordinates = eye(len(shape) + 1, dtype=get_default_dtype(), device="cpu")
        elif from_voxel_coordinates is not None and to_voxel_coordinates is not None:
            raise ValueError(
                "Only one of from_voxel_coordinates and to_voxel_coordinates should be given"
            )
        transformation_matrix = (
            to_voxel_coordinates if from_voxel_coordinates is None else from_voxel_coordinates
        )
        assert transformation_matrix is not None
        if transformation_matrix.ndim not in (2, 3):
            raise ValueError(
                "Only affine matrices (with potentially one batch dimension) are supported"
            )
        if not transformation_matrix.is_floating_point():
            raise ValueError("Affine matrices should be floating point")
        affine_transformation = CPUAffineTransformation(transformation_matrix, device=device)
        inverse_affine_transformation = affine_transformation.invert()
        self._from_voxel_coordinates, self._to_voxel_coordinates = (
            (inverse_affine_transformation, affine_transformation)
            if from_voxel_coordinates is None
            else (affine_transformation, inverse_affine_transformation)
        )
        self._shape = shape
        # Trick to make torch.nn.Module type conversion work automatically, we
        # use the empty indicator tensor to infer the device and dtype of the
        # coordinate system. Note that the actualy type conversion is done only
        # when the coordinate system is used (unless one calls
        # enforce_type_conversion())
        self.register_buffer("_indicator", transformation_matrix.new_empty(0, device=device))

    def cast(
        self,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "VoxelCoordinateSystem":
        if dtype is None and device is None:
            return self
        return VoxelCoordinateSystem(
            from_voxel_coordinates=self._from_voxel_coordinates.as_cpu_matrix().to(dtype=dtype),
            shape=self._shape,
            device=self.device if device is None else device,
        )

    def detach(self) -> "VoxelCoordinateSystem":
        return VoxelCoordinateSystem(
            from_voxel_coordinates=self._from_voxel_coordinates.as_cpu_matrix().detach(),
            shape=self._shape,
            device=self.device,
        )

    @property
    def coordinate_system(self) -> "VoxelCoordinateSystem":
        return self

    def forward(self) -> None:
        """Dummy forward pass to make the coordinate system a torch.nn.Module"""

    @property
    def dtype(self) -> torch_dtype:
        """Get dtype of the coordinate system"""
        return self.get_buffer("_indicator").dtype

    @property
    def device(self) -> torch_device:
        """Get device of the coordinate system"""
        return self.get_buffer("_indicator").device

    def enforce_type_conversion(self):
        """Enforce type conversion of the coordinate system to the current device and dtype"""
        if (
            self._from_voxel_coordinates.device != self.device
            or self._from_voxel_coordinates.dtype != self.dtype
        ):
            self._from_voxel_coordinates = self._from_voxel_coordinates.cast(
                dtype=self.dtype, device=self.device
            ).pin_memory_if_target_not_cpu()
            self._to_voxel_coordinates = self._to_voxel_coordinates.cast(
                dtype=self.dtype, device=self.device
            ).pin_memory_if_target_not_cpu()

    @property
    def from_voxel_coordinates(self) -> IComposableMapping:
        """Mapping from voxel to world coordinates"""
        self.enforce_type_conversion()
        return ComposableAffine(self._from_voxel_coordinates)

    @property
    def to_voxel_coordinates(self) -> IComposableMapping:
        """Mapping from world to voxel coordinates"""
        self.enforce_type_conversion()
        return ComposableAffine(self._to_voxel_coordinates)

    @property
    def shape(self) -> Sequence[int]:
        """Shape of the coordinate system grid"""
        return self._shape

    def grid(self) -> MappableTensor:
        """Grid in the world coordinates"""
        self.enforce_type_conversion()
        return self.from_voxel_coordinates(
            VoxelGrid(
                spatial_shape=self._shape,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def voxel_grid(self) -> MappableTensor:
        """Grid in the voxel coordinates"""
        self.enforce_type_conversion()
        return VoxelGrid(spatial_shape=self._shape, dtype=self.dtype, device=self.device)

    @staticmethod
    def _calculate_voxel_size(affine_matrix: Tensor) -> Tensor:
        return affine_matrix[..., :-1, :-1].square().sum(dim=1).sqrt()

    def grid_spacing_cpu(self) -> Tensor:
        """Get grid spacing as CPU tensor"""
        self.enforce_type_conversion()
        return self._calculate_voxel_size(self._from_voxel_coordinates.as_cpu_matrix())

    def grid_spacing(self) -> Tensor:
        """Get grid spacing"""
        self.enforce_type_conversion()
        return self._calculate_voxel_size(self._from_voxel_coordinates.as_matrix())

    def __repr__(self) -> str:
        return (
            "VoxelCoordinateSystem("
            f"from_voxel_coordinates={self.from_voxel_coordinates}, "
            f"shape={self.to_voxel_coordinates}, "
            f"device={self.device})"
        )

    def _shift(
        self, shift: Union[Sequence[Union[float, int]], float, int, Tensor], voxel: bool
    ) -> "VoxelCoordinateSystem":
        """Shift the coordinate system in the voxel coordinates

        Args:
            shift: Shift in the voxel coordinates
        """
        if not isinstance(shift, Tensor):
            shift = self._from_voxel_coordinates.as_cpu_matrix().new_tensor(shift)
        if shift.ndim > 2:
            raise ValueError("Shift should be a vector or a batch of vectors")
        n_dims = len(self._shape)
        shift = shift.expand(*shift.shape[:-1], n_dims)
        shift_matrix = generate_translation_matrix(shift)
        if voxel:
            updated_matrix = compose_affine_matrices(
                shift_matrix, self._from_voxel_coordinates.as_cpu_matrix()
            )
        else:
            updated_matrix = compose_affine_matrices(
                self._from_voxel_coordinates.as_cpu_matrix(), shift_matrix
            )
        return VoxelCoordinateSystem(
            from_voxel_coordinates=updated_matrix,
            shape=self._shape,
            device=self.device,
        )

    def shift_voxel(
        self, shift: Union[Sequence[Union[float, int]], float, int, Tensor]
    ) -> "VoxelCoordinateSystem":
        """Shift the coordinate system in the voxel coordinates

        Args:
            shift: Shift in the voxel coordinates
        """
        return self._shift(shift, voxel=True)

    def shift_world(
        self, shift: Union[Sequence[Union[float, int]], float, int, Tensor]
    ) -> "VoxelCoordinateSystem":
        """Shift the coordinate system in the world coordinates

        Args:
            shift: Shift in the world coordinates
        """
        return self._shift(shift, voxel=False)

    @overload
    def reformat(
        self,
        *,
        downsampling_factor: Optional[
            Union[Sequence[Union[float, int]], float, int, Tensor]
        ] = None,
        upsampling_factor: Optional[Union[Sequence[Union[float, int]], float, int, Tensor]] = None,
        shape: Optional[
            Union[
                Sequence[Union[IReformattingShapeOption, int]],
                IReformattingShapeOption,
                int,
                Tensor,
            ]
        ] = None,
        reference: Optional[
            Union[
                Sequence[Union[IReformattingReferenceOption, float, int]],
                IReformattingReferenceOption,
                float,
                int,
            ]
        ] = None,
        target_reference: Optional[
            Union[
                Sequence[Union[IReformattingReferenceOption, float, int]],
                IReformattingReferenceOption,
                float,
                int,
            ]
        ] = None,
    ) -> "VoxelCoordinateSystem": ...

    @overload
    def reformat(
        self,
        *,
        voxel_size: Optional[Union[Sequence[Union[float, int]], float, int, Tensor]] = None,
        shape: Optional[
            Union[
                Sequence[Union[IReformattingShapeOption, int]],
                IReformattingShapeOption,
                int,
                Tensor,
            ]
        ] = None,
        reference: Optional[
            Union[
                Sequence[Union[IReformattingReferenceOption, float, int]],
                IReformattingReferenceOption,
                float,
                int,
            ]
        ] = None,
        target_reference: Optional[
            Union[
                Sequence[Union[IReformattingReferenceOption, float, int]],
                IReformattingReferenceOption,
                float,
                int,
            ]
        ] = None,
    ) -> "VoxelCoordinateSystem": ...

    def reformat(
        self,
        *,
        downsampling_factor: Optional[
            Union[Sequence[Union[float, int]], float, int, Tensor]
        ] = None,
        upsampling_factor: Optional[Union[Sequence[Union[float, int]], float, int, Tensor]] = None,
        voxel_size: Optional[Union[Sequence[Union[float, int]], float, int, Tensor]] = None,
        shape: Optional[
            Union[
                Sequence[Union[IReformattingShapeOption, int]],
                IReformattingShapeOption,
                int,
                Tensor,
            ]
        ] = None,
        reference: Optional[
            Union[
                Sequence[Union[IReformattingReferenceOption, float, int]],
                IReformattingReferenceOption,
                float,
                int,
            ]
        ] = None,
        target_reference: Optional[
            Union[
                Sequence[Union[IReformattingReferenceOption, float, int]],
                IReformattingReferenceOption,
                float,
                int,
            ]
        ] = None,
    ) -> "VoxelCoordinateSystem":
        """Reformat the coordinate system

        Note that voxel_size can not be used together with downsampling_factor
        and upsampling_factor

        Args:
            downsampling_factor: Factor to downsample the grid voxel size, if given
                as a tensor, it should be on CPU, and have the same dtype as the
                coordinate system.
            upsampling_factor: Factor to upsample the grid voxel size, if given
                as a tensor, it should be on CPU, and have the same dtype as the
                coordinate system.
            voxel_size: Voxel size of the grid, if given
                as a tensor, it should be on CPU, and have the same dtype as the
                coordinate system.
            shape: Defines shape of the target grid, either given separately for
                each dimension or as a single value in which case the same value
                is used for all the dimensions. Defaults to
                FitToFOVOption("round", "full_voxels")
            reference: Defines the point in the original voxel
                coordinates which will be aligned with the target reference in
                the reformatted coordinates. Either given separately for each
                dimension or as a single value in which case the same value is
                used for all the dimensions. Defaults to ReferenceOption("left",
                "full_voxels")
            target_reference: Defaults to reference. Defines the point in the
                reformatted voxel coordinates which will be aligned with the
                source reference in the original coordinates. Either given
                separately for each dimension or as a single value in which case
                the same value is used for all the dimensions.
        """
        original_voxel_size = self.grid_spacing_cpu()
        downsampling_factor = self._as_downsampling_factor(
            original_voxel_size, downsampling_factor, upsampling_factor, voxel_size
        )
        if shape is None:
            shape = FitToFOVOption(fitting_method="round", fov_convention="full_voxels")
        target_shape = self._get_target_shape(downsampling_factor, shape)
        if reference is None:
            reference = ReferenceOption(position="left", fov_convention="full_voxels")
        source_reference_in_voxel_coordinates = original_voxel_size.new_tensor(
            self._get_reference_in_voxel_coordinates(reference, self._shape)
        )
        if target_reference is None:
            target_reference = reference
        target_reference_in_voxel_coordinates = original_voxel_size.new_tensor(
            self._get_reference_in_voxel_coordinates(target_reference, target_shape)
        )
        source_translation_matrix = generate_translation_matrix(
            source_reference_in_voxel_coordinates
        )
        target_translation_matrix = generate_translation_matrix(
            -target_reference_in_voxel_coordinates
        )
        n_dims = original_voxel_size.size(-1)
        downsampling_matrix = embed_matrix(
            generate_scale_matrix(downsampling_factor), (n_dims + 1, n_dims + 1)
        )
        reformatted_transformation = compose_affine_matrices(
            self._from_voxel_coordinates.as_cpu_matrix(),
            source_translation_matrix,
            downsampling_matrix,
            target_translation_matrix,
        )

        return VoxelCoordinateSystem(
            from_voxel_coordinates=reformatted_transformation,
            shape=target_shape,
            device=self.device,
        )

    def _as_downsampling_factor(
        self,
        original_voxel_size: Tensor,
        downsampling_factor: Optional[
            Union[Sequence[Union[float, int]], float, int, Tensor]
        ] = None,
        upsampling_factor: Optional[Union[Sequence[Union[float, int]], float, int, Tensor]] = None,
        voxel_size: Optional[Union[Sequence[Union[int, float]], float, int, Tensor]] = None,
    ) -> Tensor:
        if voxel_size is not None:
            if not isinstance(voxel_size, Tensor):
                voxel_size = original_voxel_size.new_tensor(voxel_size)
            if not voxel_size.is_floating_point() or voxel_size.device != torch_device("cpu"):
                raise ValueError("Voxel size should be a CPU floating point tensor")
            return voxel_size / original_voxel_size
        if downsampling_factor is not None:
            if not isinstance(downsampling_factor, Tensor):
                downsampling_factor = original_voxel_size.new_tensor(downsampling_factor)
            if (
                not downsampling_factor.is_floating_point()
                or downsampling_factor.device != torch_device("cpu")
            ):
                raise ValueError("Downsampling factor should be a CPU floating point tensor")
            processed_downsampling_factor = downsampling_factor
        else:
            processed_downsampling_factor = original_voxel_size.new_ones(1)
        if upsampling_factor is not None:
            if not isinstance(upsampling_factor, Tensor):
                upsampling_factor = original_voxel_size.new_tensor(upsampling_factor)
            if (
                not upsampling_factor.is_floating_point()
                or upsampling_factor.device != torch_device("cpu")
            ):
                raise ValueError("Upsampling factor should be a CPU floating point tensor")
            processed_upsampling_factor = 1 / upsampling_factor
        else:
            processed_upsampling_factor = original_voxel_size.new_ones(1)
        return (
            processed_downsampling_factor
            * processed_upsampling_factor
            * ones_like(original_voxel_size)
        )

    def _get_reference_in_voxel_coordinates(
        self,
        reference: Union[
            Sequence[Union[IReformattingReferenceOption, float, int]],
            IReformattingReferenceOption,
            float,
            int,
        ],
        shape: Sequence[int],
    ) -> List[float]:
        if isinstance(reference, (IReformattingReferenceOption, float, int)):
            reference = [reference] * len(shape)
        voxel_coordinate_reference: List[float] = []
        for dim_reference, dim_size in zip(reference, shape):
            if isinstance(dim_reference, (float, int)):
                voxel_coordinate_reference.append(dim_reference)
            else:
                voxel_coordinate_reference.append(dim_reference.get_voxel_coordinate(dim_size))
        return voxel_coordinate_reference

    def _get_target_shape(
        self,
        downsampling_factor: Tensor,
        shape: Union[
            Sequence[Union[IReformattingShapeOption, int]], IReformattingShapeOption, int, Tensor
        ],
    ) -> Sequence[int]:
        if isinstance(shape, Tensor):
            if shape.ndim not in (0, 1):
                raise ValueError("Invalid shape tensor for reformatting")
            if shape.is_floating_point() or shape.is_complex():
                raise ValueError("Shape tensor should be an integer tensor")
            shape = shape.expand((len(self._shape),))
            shape = shape.tolist()
        if isinstance(shape, (int, IReformattingShapeOption)):
            shape = [shape] * len(self._shape)
        if downsampling_factor.ndim == 1:
            downsampling_factor = downsampling_factor.unsqueeze(0)
        output_shape: Optional[Sequence[int]] = None
        for batch_index in range(downsampling_factor.size(0)):
            target_shape = self._get_target_shape_for_batch(downsampling_factor[batch_index], shape)
            if output_shape is None:
                output_shape = target_shape
            else:
                if output_shape != target_shape:
                    raise ValueError(
                        "Inconsistent target shapes obtained with given shape options, "
                        "consider defining an explicit target shape for reformatting."
                    )
        assert output_shape is not None
        return output_shape

    def _get_target_shape_for_batch(
        self,
        single_downsampling_factor: Tensor,
        shape: Sequence[Union[IReformattingShapeOption, int]],
    ) -> Sequence[int]:
        target_shape = []
        for dim_original_shape, dim_shape, dim_downsampling_factor in zip(
            self._shape, shape, single_downsampling_factor
        ):
            if isinstance(dim_shape, IReformattingShapeOption):
                target_shape.append(
                    dim_shape.to_target_size(dim_original_shape, dim_downsampling_factor.item())
                )
            elif isinstance(dim_shape, int):
                target_shape.append(dim_shape)
            else:
                raise ValueError(f"Invalid shape for reformatting: {shape}")
        return target_shape


def create_centered_normalized(
    shape: Sequence[int],
    voxel_size: Union[Sequence[Union[float, int]], float, int, Tensor] = 1.0,
    align_corners: bool = False,
    device: Optional[torch_device] = None,
) -> VoxelCoordinateSystem:
    """Create normalized coordinate system with the given shape and device"""
    centered = create_centered(shape, voxel_size, device)
    voxel_size = centered.grid_spacing_cpu()
    shape_tensor = voxel_size.new_tensor(shape)
    if align_corners:
        shape_tensor -= 1
    fov_sizes = shape_tensor * voxel_size
    max_fov_size = fov_sizes.amax(dim=-1)
    target_voxel_size = 2 / max_fov_size
    return centered.reformat(
        voxel_size=target_voxel_size,
        reference=ReferenceOption(position="center"),
        shape=RetainShapeOption(),
    )


def create_centered(
    shape: Sequence[int],
    voxel_size: Union[Sequence[Union[float, int]], float, int, Tensor] = 1.0,
    device: Optional[torch_device] = None,
) -> VoxelCoordinateSystem:
    """Create centered coordinate system"""
    return create_voxel(shape, voxel_size=voxel_size, device=device).shift_voxel(
        [-(dim_size - 1) / 2 for dim_size in shape]
    )


def create_voxel(
    shape: Sequence[int],
    voxel_size: Union[Sequence[Union[float, int]], float, int, Tensor] = 1.0,
    device: Optional[torch_device] = None,
) -> VoxelCoordinateSystem:
    """Create voxel coordinate system"""
    n_dims = len(shape)
    if isinstance(voxel_size, (float, int)):
        voxel_size = [voxel_size] * n_dims
    if not isinstance(voxel_size, Tensor):
        voxel_size = tensor(voxel_size, dtype=get_default_dtype(), device="cpu")
    if voxel_size.size(-1) != n_dims:
        raise ValueError("Invalid voxel size for the coordinate system")
    if not voxel_size.is_floating_point():
        raise ValueError("Voxel size should be a floating point tensor")
    if voxel_size.ndim > 2:
        raise ValueError("Voxel size should be a vector or a batch of vectors")
    initial_affine = embed_matrix(generate_scale_matrix(voxel_size), (n_dims + 1, n_dims + 1))
    return VoxelCoordinateSystem(
        from_voxel_coordinates=initial_affine,
        shape=shape,
        device=device,
    )
