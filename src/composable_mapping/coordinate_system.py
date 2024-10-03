"""Voxel coordinate system"""

from abc import ABC, abstractmethod
from typing import List, Mapping, Optional, Sequence, Union, cast, overload

from torch import Tensor
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import empty, get_default_dtype, tensor
from torch.nn import Module

from .affine import Affine
from .mappable_tensor import (
    HostDiagonalAffineTransformation,
    IHostAffineTransformation,
    MappableTensor,
    VoxelGrid,
)
from .tensor_like import BaseTensorLikeWrapper, ITensorLike
from .util import (
    broadcast_shapes_in_parts_splitted,
    broadcast_tensors_in_parts,
    broadcast_to_in_parts,
    ceildiv,
    get_channel_dims,
    get_channels_shape,
    get_spatial_shape,
    move_channels_last,
)


class ICoordinateSystemContainer(ABC):
    """Class holding a unique voxel coordinate system"""

    @property
    @abstractmethod
    def coordinate_system(
        self,
    ) -> "CoordinateSystem":
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


class CoordinateSystem(Module, ICoordinateSystemContainer, BaseTensorLikeWrapper):
    """Represents coordinate system between voxel and world coordinates

    Arguments:
        shape: Shape of the grid
        to_voxel_coordinates: Affine transformation from world to voxel coordinates,
            should be inverse of from_voxel_coordinates, can be omitted if
            from_voxel_coordinates is given
        from_voxel_coordinates: Affine transformation from voxel to world coordinates,
            should be inverse of to_voxel_coordinates, can be omitted if
            to_voxel_coordinates is given
    """

    def __init__(
        self,
        shape: Sequence[int],
        to_voxel_coordinates: Optional[IHostAffineTransformation] = None,
        from_voxel_coordinates: Optional[IHostAffineTransformation] = None,
    ) -> None:
        super().__init__()
        self._shape = shape
        if from_voxel_coordinates is None:
            if to_voxel_coordinates is None:
                raise ValueError(
                    "Either from_voxel_coordinates or to_voxel_coordinates should be given"
                )
            from_voxel_coordinates = to_voxel_coordinates.invert()
        elif to_voxel_coordinates is None:
            to_voxel_coordinates = from_voxel_coordinates.invert()
        self._coordinate_transformations_which_should_not_be_accessed_directly = {
            "from_voxel_coordinates": from_voxel_coordinates,
            "to_voxel_coordinates": to_voxel_coordinates,
        }
        if (
            get_channels_shape(from_voxel_coordinates.matrix_shape, n_channel_dims=2)
            != (
                len(shape) + 1,
                len(shape) + 1,
            )
            or get_spatial_shape(from_voxel_coordinates.matrix_shape, n_channel_dims=2) != tuple()
        ):
            raise ValueError("Invalid affine transformation for a coordinate system")
        if (
            get_channels_shape(to_voxel_coordinates.matrix_shape, n_channel_dims=2)
            != (
                len(shape) + 1,
                len(shape) + 1,
            )
            or get_spatial_shape(to_voxel_coordinates.matrix_shape, n_channel_dims=2) != tuple()
        ):
            raise ValueError("Invalid affine transformation for a coordinate system")
        # Trick to make torch.nn.Module type conversion work automatically, we
        # use the empty indicator tensor to infer the device and dtype of the
        # coordinate system.
        self.register_buffer(
            "_indicator",
            empty(0, device=to_voxel_coordinates.device, dtype=to_voxel_coordinates.dtype),
        )

    @classmethod
    def centered_normalized(
        cls,
        shape: Sequence[int],
        voxel_size: Union[Sequence[Number], Number, Tensor] = 1.0,
        align_corners: bool = False,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "CoordinateSystem":
        """Create centered normalized coordinate system

        The coordinate system is normalized such that the grid just fits into the
        cube from -1 to 1 in each dimension. The grid is centered in the cube.

        Args:
            shape: Shape of the grid
            voxel_size: Voxel size of the grid
            align_corners: Whether to fit the grid into the cube by full voxels or
                by voxel centers (see similar option in torch.nn.functional.grid_sample)
            device: Device of the coordinate system
        """
        centered = cls.centered(shape, voxel_size, dtype=dtype, device=device)
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

    @classmethod
    def centered(
        cls,
        shape: Sequence[int],
        voxel_size: Union[Sequence[Number], Number, Tensor] = 1.0,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "CoordinateSystem":
        """Create centered coordinate system with given voxel size

        Args:
            shape: Shape of the grid
            voxel_size: Voxel size of the grid
            device: Device of the coordinate system
        """
        return cls.voxel(shape, voxel_size=voxel_size, dtype=dtype, device=device).shift_voxel(
            [-(dim_size - 1) / 2 for dim_size in shape]
        )

    @classmethod
    def voxel(
        cls,
        shape: Sequence[int],
        voxel_size: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        dtype: Optional[torch_dtype] = None,
        device: Optional[torch_device] = None,
    ) -> "CoordinateSystem":
        """Create coordinate system corresponding to the voxel coordinates with
        potential scaling by voxel size"""
        n_dims = len(shape)
        if dtype is None:
            dtype = get_default_dtype()
        if voxel_size is not None:
            if isinstance(voxel_size, (float, int)):
                voxel_size = [voxel_size] * n_dims
            if isinstance(voxel_size, Tensor):
                if dtype is not None and voxel_size.dtype != dtype:
                    raise ValueError(
                        "Data type of the voxel size should match the given data type, "
                        "note that the dtype option can be omitted when voxel size "
                        "is given as a tensor."
                    )
            else:
                voxel_size = tensor(voxel_size, dtype=dtype, device=torch_device("cpu"))
            if get_channels_shape(voxel_size.shape, n_channel_dims=1)[0] != n_dims:
                raise ValueError("Invalid voxel size for the coordinate system")
        return cls(
            from_voxel_coordinates=HostDiagonalAffineTransformation(
                diagonal=voxel_size, matrix_shape=(n_dims + 1, n_dims + 1), device=device
            ),
            shape=shape,
        )

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        return self

    def forward(self) -> None:
        """Dummy forward pass to make the coordinate system a torch.nn.Module"""

    @property
    def _from_voxel_coordinates(self) -> IHostAffineTransformation:
        from_voxel_coordinates = (
            self._coordinate_transformations_which_should_not_be_accessed_directly[
                "from_voxel_coordinates"
            ]
        )
        if (
            from_voxel_coordinates.device != self.device
            or from_voxel_coordinates.dtype != self.dtype
        ):
            self._coordinate_transformations_which_should_not_be_accessed_directly[
                "from_voxel_coordinates"
            ] = from_voxel_coordinates.cast(dtype=self.dtype, device=self.device)
        return self._coordinate_transformations_which_should_not_be_accessed_directly[
            "from_voxel_coordinates"
        ]

    @property
    def _to_voxel_coordinates(self) -> IHostAffineTransformation:
        to_voxel_coordinates = (
            self._coordinate_transformations_which_should_not_be_accessed_directly[
                "to_voxel_coordinates"
            ]
        )
        if to_voxel_coordinates.device != self.device or to_voxel_coordinates.dtype != self.dtype:
            self._coordinate_transformations_which_should_not_be_accessed_directly[
                "to_voxel_coordinates"
            ] = to_voxel_coordinates.cast(dtype=self.dtype, device=self.device)
        return self._coordinate_transformations_which_should_not_be_accessed_directly[
            "to_voxel_coordinates"
        ]

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {
            "from_voxel_coordinates": self._from_voxel_coordinates,
            "to_voxel_coordinates": self._to_voxel_coordinates,
        }

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "CoordinateSystem":
        if not isinstance(children["from_voxel_coordinates"], IHostAffineTransformation):
            raise ValueError("from_voxel_coordinates should be an affine transformation")
        if not isinstance(children["to_voxel_coordinates"], IHostAffineTransformation):
            raise ValueError("to_voxel_coordinates should be an affine transformation")
        return CoordinateSystem(
            shape=self._shape,
            from_voxel_coordinates=children["from_voxel_coordinates"],
            to_voxel_coordinates=children["to_voxel_coordinates"],
        )

    @property
    def dtype(self) -> torch_dtype:
        """Get dtype of the coordinate system"""
        return self.get_buffer("_indicator").dtype

    @property
    def device(self) -> torch_device:
        """Get device of the coordinate system"""
        return self.get_buffer("_indicator").device

    @property
    def from_voxel_coordinates(self) -> Affine:
        """Mapping from voxel to world coordinates"""
        return Affine(self._from_voxel_coordinates)

    @property
    def to_voxel_coordinates(self) -> Affine:
        """Mapping from world to voxel coordinates"""
        return Affine(self._to_voxel_coordinates)

    @property
    def shape(self) -> Sequence[int]:
        """Shape of the coordinate system grid"""
        return self._shape

    def grid(self) -> MappableTensor:
        """Grid in the world coordinates"""
        return self.from_voxel_coordinates(
            VoxelGrid(
                spatial_shape=self._shape,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def voxel_grid(self) -> MappableTensor:
        """Grid in the voxel coordinates"""
        return VoxelGrid(spatial_shape=self._shape, dtype=self.dtype, device=self.device)

    @staticmethod
    def _calculate_voxel_size(affine_matrix: Tensor) -> Tensor:
        channel_dims = get_channel_dims(affine_matrix.ndim, n_channel_dims=2)
        row_dim = channel_dims[0]
        col_dim = channel_dims[1]
        matrix = affine_matrix.narrow(row_dim, 0, affine_matrix.size(row_dim) - 1)
        matrix = matrix.narrow(col_dim, 0, affine_matrix.size(col_dim) - 1)
        return matrix.square().sum(dim=row_dim).sqrt()

    def grid_spacing_cpu(self) -> Tensor:
        """Get grid spacing as CPU tensor"""
        diagonal_on_host = self._from_voxel_coordinates.as_host_diagonal()
        if diagonal_on_host is None:
            return self._calculate_voxel_size(self._from_voxel_coordinates.as_host_matrix())
        return diagonal_on_host.generate_diagonal().abs()

    def grid_spacing(self) -> Tensor:
        """Get grid spacing"""
        diagonal = self._from_voxel_coordinates.as_diagonal()
        if diagonal is None:
            return self._calculate_voxel_size(self._from_voxel_coordinates.as_matrix())
        return diagonal.generate_diagonal().abs()

    def __repr__(self) -> str:
        return (
            "CoordinateSystem("
            f"shape={self._shape}, "
            f"to_voxel_coordinates={self._to_voxel_coordinates}, "
            f"from_voxel_coordinates={self.from_voxel_coordinates})"
        )

    def _shift(
        self, shift: Union[Sequence[Number], Number, Tensor], in_voxel_coordinates: bool
    ) -> "CoordinateSystem":
        """Shift the coordinate system in the voxel coordinates

        Args:
            shift: Shift in the voxel coordinates
        """
        if not isinstance(shift, Tensor):
            shift = tensor(shift, dtype=self.dtype, device=torch_device("cpu"))
        shift = broadcast_to_in_parts(shift, channels_shape=(len(self._shape),), n_channel_dims=1)
        shift_transformation = HostDiagonalAffineTransformation(
            translation=shift, device=self.device
        )
        if in_voxel_coordinates:
            updated_transformation = shift_transformation @ self._from_voxel_coordinates
        else:
            updated_transformation = self._from_voxel_coordinates @ shift_transformation
        return CoordinateSystem(
            from_voxel_coordinates=updated_transformation,
            shape=self._shape,
        )

    def shift_voxel(self, shift: Union[Sequence[Number], Number, Tensor]) -> "CoordinateSystem":
        """Shift the coordinate system in the voxel coordinates

        Args:
            shift: Shift in the voxel coordinates
        """
        return self._shift(shift, in_voxel_coordinates=True)

    def shift_world(self, shift: Union[Sequence[Number], Number, Tensor]) -> "CoordinateSystem":
        """Shift the coordinate system in the world coordinates

        Args:
            shift: Shift in the world coordinates
        """
        return self._shift(shift, in_voxel_coordinates=False)

    @overload
    def reformat(
        self,
        *,
        downsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        upsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
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
                Sequence[Union[IReformattingReferenceOption, Number]],
                IReformattingReferenceOption,
                Number,
            ]
        ] = None,
        target_reference: Optional[
            Union[
                Sequence[Union[IReformattingReferenceOption, Number]],
                IReformattingReferenceOption,
                Number,
            ]
        ] = None,
    ) -> "CoordinateSystem": ...

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
                Sequence[Union[IReformattingReferenceOption, Number]],
                IReformattingReferenceOption,
                Number,
            ]
        ] = None,
        target_reference: Optional[
            Union[
                Sequence[Union[IReformattingReferenceOption, Number]],
                IReformattingReferenceOption,
                Number,
            ]
        ] = None,
    ) -> "CoordinateSystem": ...

    def reformat(
        self,
        *,
        downsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        upsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        voxel_size: Optional[Union[Sequence[Number], Number, Tensor]] = None,
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
                Sequence[Union[IReformattingReferenceOption, Number]],
                IReformattingReferenceOption,
                Number,
            ]
        ] = None,
        target_reference: Optional[
            Union[
                Sequence[Union[IReformattingReferenceOption, Number]],
                IReformattingReferenceOption,
                Number,
            ]
        ] = None,
    ) -> "CoordinateSystem":
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
        source_translation = HostDiagonalAffineTransformation(
            translation=source_reference_in_voxel_coordinates, device=self.device
        )
        target_translation = HostDiagonalAffineTransformation(
            translation=-target_reference_in_voxel_coordinates, device=self.device
        )
        downsampling_translation = HostDiagonalAffineTransformation(
            downsampling_factor, device=self.device
        )
        reformatted_transformation = (
            self._from_voxel_coordinates
            @ source_translation
            @ downsampling_translation
            @ target_translation
        )

        return CoordinateSystem(
            from_voxel_coordinates=reformatted_transformation,
            shape=target_shape,
        )

    def _as_downsampling_factor(
        self,
        original_voxel_size: Tensor,
        downsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        upsampling_factor: Optional[Union[Sequence[Number], Number, Tensor]] = None,
        voxel_size: Optional[Union[Sequence[Number], Number, Tensor]] = None,
    ) -> Tensor:
        if voxel_size is not None:
            if not isinstance(voxel_size, Tensor):
                voxel_size = original_voxel_size.new_tensor(voxel_size)
            voxel_size, original_voxel_size = broadcast_tensors_in_parts(
                voxel_size, original_voxel_size
            )
            if get_spatial_shape(voxel_size.shape, n_channel_dims=1) != tuple():
                raise ValueError(
                    "Downsampling factors or upsampling factors should not have spatial dimensions"
                )
            return voxel_size / original_voxel_size
        if downsampling_factor is not None:
            if not isinstance(downsampling_factor, Tensor):
                downsampling_factor = original_voxel_size.new_tensor(downsampling_factor)
            processed_downsampling_factor = downsampling_factor
        else:
            processed_downsampling_factor = original_voxel_size.new_ones(1)
        if upsampling_factor is not None:
            if not isinstance(upsampling_factor, Tensor):
                upsampling_factor = original_voxel_size.new_tensor(upsampling_factor)
            processed_upsampling_factor = 1 / upsampling_factor
        else:
            processed_upsampling_factor = original_voxel_size.new_ones(1)
        _batch_shape, channels_shape, spatial_shape = broadcast_shapes_in_parts_splitted(
            processed_downsampling_factor.shape,
            processed_upsampling_factor.shape,
            original_voxel_size.shape,
        )
        if spatial_shape != tuple():
            raise ValueError(
                "Downsampling factors or upsampling factors should not have spatial dimensions"
            )
        processed_downsampling_factor = broadcast_to_in_parts(
            processed_downsampling_factor,
            channels_shape=channels_shape,
        )
        processed_upsampling_factor = broadcast_to_in_parts(
            processed_upsampling_factor,
            channels_shape=channels_shape,
        )
        return processed_downsampling_factor * processed_upsampling_factor

    def _get_reference_in_voxel_coordinates(
        self,
        reference: Union[
            Sequence[Union[IReformattingReferenceOption, Number]],
            IReformattingReferenceOption,
            Number,
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
            return shape.tolist()
        if isinstance(shape, int):
            return [shape] * len(self._shape)
        if isinstance(shape, IReformattingShapeOption):
            shape = [shape] * len(self._shape)
        else:
            if len(shape) != len(self._shape):
                raise ValueError("Invalid shape for reformatting")
            if all(isinstance(dim_shape, int) for dim_shape in shape):
                return cast(Sequence[int], shape)
        downsampling_factor = move_channels_last(downsampling_factor).view(
            -1, get_channels_shape(downsampling_factor.shape, n_channel_dims=1)[0]
        )
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
