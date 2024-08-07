"""Voxel coordinate system"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union, overload

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import ones_like
from torch.nn import Module

from .affine import (
    CPUAffineTransformation,
    compose_affine_transformation_matrices,
    embed_transformation,
    generate_scale_matrix,
    generate_translation_matrix,
)
from .ceildiv import ceildiv
from .interface import IAffineTransformation, IMaskedTensor
from .masked_tensor import VoxelCoordinateGrid


class IVoxelCoordinateSystemContainer(ABC):
    """Class holding a unique voxel coordinate system apart from its dtype and device"""

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
        "floor": lambda x, y: x // y,
        "ceil": ceildiv,
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
        fov_convention: str = "square_voxels",
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
        if self._fov_convention == "squares":
            if self._position == "left":
                return -0.5
            if self._position == "right":
                return size - 0.5
        elif self._fov_convention == "centers":
            if self._position == "left":
                return 0
            if self._position == "right":
                return size - 1
        raise ValueError(f"Unknown position {self._position}")


class VoxelCoordinateSystem(Module, IVoxelCoordinateSystemContainer):
    """Represents coordinate system between voxel and world coordinates

    Arguments:
        from_voxel_coordinates: Affine transformation from voxel to world coordinates,
            the tensor should be on CPU
        shape: Shape of the grid in the voxel coordinates
        device: Device of the coordinate system
    """

    def __init__(
        self,
        from_voxel_coordinates: Tensor,
        shape: Sequence[int],
        device: Optional[torch_device],
    ):
        super().__init__()
        if from_voxel_coordinates.ndim not in (2, 3):
            raise ValueError(
                "Only affine matrices (with potentially one batch dimension) are supported"
            )
        if not from_voxel_coordinates.is_floating_point():
            raise ValueError("Affine matrices should be floating point")
        self._from_voxel_coordinates = CPUAffineTransformation(
            from_voxel_coordinates, device=device
        ).pin_memory_if_needed()
        self._to_voxel_coordinates = (
            self._from_voxel_coordinates.invert().reduce().pin_memory_if_needed()
        )
        self._shape = shape
        # Trick to make torch.nn.Module type conversion work automatically, we
        # use the empty indicator tensor to infer the device and dtype of the
        # coordinate system. Note that the actualy type conversion is done only
        # when the coordinate system is used (unless one calls
        # enforce_type_conversion())
        self.register_buffer("_indicator", from_voxel_coordinates.new_empty(0))

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
            self._from_voxel_coordinates = self._from_voxel_coordinates.to(
                dtype=self.dtype, device=self.device
            ).pin_memory_if_needed()
            self._to_voxel_coordinates = self._to_voxel_coordinates.to(
                dtype=self.dtype, device=self.device
            ).pin_memory_if_needed()

    @property
    def from_voxel_coordinates(self) -> IAffineTransformation:
        """Mapping from voxel to world coordinates"""
        self.enforce_type_conversion()
        return self._from_voxel_coordinates

    @property
    def to_voxel_coordinates(self) -> IAffineTransformation:
        """Mapping from world to voxel coordinates"""
        self.enforce_type_conversion()
        return self._to_voxel_coordinates

    @property
    def shape(self) -> Sequence[int]:
        """Shape of the coordinate system grid"""
        return self._shape

    def grid(self) -> VoxelCoordinateGrid:
        """Grid in the world coordinates"""
        self.enforce_type_conversion()
        return VoxelCoordinateGrid(
            shape=self._shape,
            affine_transformation=self.from_voxel_coordinates,
            dtype=self.dtype,
            device=self.device,
        )

    def voxel_grid(self) -> IMaskedTensor:
        """Grid in the voxel coordinates"""
        self.enforce_type_conversion()
        return VoxelCoordinateGrid(shape=self._shape, dtype=self.dtype, device=self.device)

    @staticmethod
    def _calculate_voxel_size(affine_matrix: Tensor) -> Tensor:
        return affine_matrix[:, :-1, :-1].square().sum(dim=1).sqrt()

    def grid_spacing(self) -> Tensor:
        """Get grid spacing as CPU tensor"""
        self.enforce_type_conversion()
        return self._calculate_voxel_size(self._from_voxel_coordinates.as_cpu_matrix())

    def __repr__(self) -> str:
        return (
            "VoxelCoordinateSystem("
            f"from_voxel_coordinates={self.from_voxel_coordinates}, "
            f"shape={self.to_voxel_coordinates}, "
            f"device={self.device})"
        )

    @overload
    def reformat(
        self,
        *,
        downsampling_factor: Optional[
            Union[Sequence[Union[float, int]], float, int, Tensor]
        ] = None,
        shape: Optional[
            Union[Sequence[Union[IReformattingShapeOption, int]], IReformattingShapeOption, int]
        ] = None,
        source_reference: Optional[
            Union[
                Sequence[IReformattingReferenceOption],
                IReformattingReferenceOption,
            ]
        ] = None,
        target_reference: Optional[
            Union[
                Sequence[IReformattingReferenceOption],
                IReformattingReferenceOption,
            ]
        ] = None,
    ) -> "VoxelCoordinateSystem": ...

    @overload
    def reformat(
        self,
        *,
        upsampling_factor: Optional[Union[Sequence[Union[float, int]], float, int, Tensor]] = None,
        shape: Optional[
            Union[Sequence[Union[IReformattingShapeOption, int]], IReformattingShapeOption, int]
        ] = None,
        source_reference: Optional[
            Union[
                Sequence[IReformattingReferenceOption],
                IReformattingReferenceOption,
            ]
        ] = None,
        target_reference: Optional[
            Union[
                Sequence[IReformattingReferenceOption],
                IReformattingReferenceOption,
            ]
        ] = None,
    ) -> "VoxelCoordinateSystem": ...

    @overload
    def reformat(
        self,
        *,
        voxel_size: Optional[Union[Sequence[Union[float, int]], float, int, Tensor]] = None,
        shape: Optional[
            Union[Sequence[Union[IReformattingShapeOption, int]], IReformattingShapeOption, int]
        ] = None,
        source_reference: Optional[
            Union[
                Sequence[IReformattingReferenceOption],
                IReformattingReferenceOption,
            ]
        ] = None,
        target_reference: Optional[
            Union[
                Sequence[IReformattingReferenceOption],
                IReformattingReferenceOption,
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
        source_reference: Optional[
            Union[
                Sequence[IReformattingReferenceOption],
                IReformattingReferenceOption,
            ]
        ] = None,
        target_reference: Optional[
            Union[
                Sequence[IReformattingReferenceOption],
                IReformattingReferenceOption,
            ]
        ] = None,
    ) -> "VoxelCoordinateSystem":
        """Reformat the coordinate system

        Provide only one of the parameters: downsampling_factor, upsampling_factor, voxel_size

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
            source_reference: Defines the point in the original coordinates
                which will be aligned with the target reference in the
                reformatted coordinates. Either given separately for each
                dimension or as a single value in which case the same value is
                used for all the dimensions. Defaults to ReferenceOption("left",
                "full_voxels")
            target_reference: Defines the point in the reformatted coordinates
                which will be aligned with the source reference in the original
                coordinates. Either given separately for each dimension or as a
                single value in which case the same value is used for all the
                dimensions. Defaults to source_reference.
        """
        original_voxel_size = self.grid_spacing()
        downsampling_factor = self._as_downsampling_factor(
            original_voxel_size, downsampling_factor, upsampling_factor, voxel_size
        )
        if shape is None:
            shape = FitToFOVOption(fitting_method="round", fov_convention="full_voxels")
        target_shape = self._get_target_shape(downsampling_factor, shape)
        if source_reference is None:
            source_reference = ReferenceOption(position="left", fov_convention="square_voxels")
        source_reference_in_voxel_coordinates = original_voxel_size.new(
            self._get_reference_in_voxel_coordinates(source_reference, self._shape)
        )
        if target_reference is None:
            target_reference = source_reference
        target_reference_in_voxel_coordinates = original_voxel_size.new(
            self._get_reference_in_voxel_coordinates(target_reference, target_shape)
        )
        source_translation_matrix = generate_translation_matrix(
            -source_reference_in_voxel_coordinates
        )
        target_translation_matrix = generate_translation_matrix(
            target_reference_in_voxel_coordinates
        )
        n_dims = original_voxel_size.size(-1)
        downsampling_matrix = embed_transformation(
            generate_scale_matrix(downsampling_factor), (n_dims + 1, n_dims + 1)
        )
        reformatted_transformation = compose_affine_transformation_matrices(
            target_translation_matrix,
            downsampling_matrix,
            self._from_voxel_coordinates.as_cpu_matrix(),
            source_translation_matrix,
        )

        return VoxelCoordinateSystem(
            from_voxel_coordinates=reformatted_transformation,
            shape=self._shape,
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
        if downsampling_factor is not None:
            if upsampling_factor is not None or voxel_size is not None:
                raise ValueError(
                    "Can not specify downsampling factor together with upsampling factor or "
                    "voxel size"
                )
            if not isinstance(downsampling_factor, Tensor):
                downsampling_factor = original_voxel_size.new(downsampling_factor)
            if (
                not downsampling_factor.is_floating_point()
                or downsampling_factor.device != torch_device("cpu")
            ):
                raise ValueError("Downsampling factor should be a CPU floating point tensor")
            return downsampling_factor.broadcast_to(original_voxel_size.shape)
        if upsampling_factor is not None:
            if voxel_size is not None:
                raise ValueError("Can not specify upsampling factor together with voxel size")
            if not isinstance(upsampling_factor, Tensor):
                upsampling_factor = original_voxel_size.new(upsampling_factor)
            if (
                not upsampling_factor.is_floating_point()
                or upsampling_factor.device != torch_device("cpu")
            ):
                raise ValueError("Upsampling factor should be a CPU floating point tensor")
            return (1 / upsampling_factor).broadcast_to(original_voxel_size.shape)
        if voxel_size is not None:
            if isinstance(voxel_size, (float, int)):
                voxel_size = [voxel_size] * original_voxel_size.size(-1)
            if not isinstance(voxel_size, Tensor):
                voxel_size = original_voxel_size.new(voxel_size)
            if not voxel_size.is_floating_point() or voxel_size.device != torch_device("cpu"):
                raise ValueError("Voxel size should be a CPU floating point tensor")
            return voxel_size / original_voxel_size
        return ones_like(original_voxel_size)

    def _get_reference_in_voxel_coordinates(
        self,
        reference: Union[
            Sequence[IReformattingReferenceOption],
            IReformattingReferenceOption,
        ],
        shape: Sequence[int],
    ) -> List[float]:
        if isinstance(reference, IReformattingReferenceOption):
            reference = [reference] * len(shape)
        voxel_coordinate_reference: List[float] = []
        for dim_reference, dim_size in zip(reference, shape):
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
