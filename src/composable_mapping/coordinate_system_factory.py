"""Factory functions for creating coordinate systems"""

from typing import Optional, Sequence, Tuple

from torch import Tensor
from torch import device as torch_device
from torch import diag
from torch import dtype as torch_dtype
from torch import tensor

from .affine import ComposableAffine, CPUAffineTransformation
from .interface import IVoxelCoordinateSystemFactory
from .masked_tensor import VoxelCoordinateGrid
from .voxel_coordinate_system import VoxelCoordinateSystem


class CenteredNormalizedFactory(IVoxelCoordinateSystemFactory):
    """Factory for creating centered normalized coordinate systems"""

    def __init__(
        self,
        original_grid_shape: Sequence[int],
        original_voxel_size: Optional[Sequence[float]] = None,
        grid_shape: Optional[Sequence[int]] = None,
        voxel_size: Optional[Sequence[float]] = None,
        downsampling_factor: Optional[Sequence[float]] = None,
        center_coordinate: Optional[Sequence[float]] = None,
    ):
        self.original_grid_shape = original_grid_shape
        self.original_voxel_size = original_voxel_size
        self.grid_shape = grid_shape
        self.voxel_size = voxel_size
        self.downsampling_factor = downsampling_factor
        self.center_coordinate = center_coordinate

    def create(self, dtype: torch_dtype, device: torch_device) -> VoxelCoordinateSystem:
        return create_centered_normalized(
            original_grid_shape=self.original_grid_shape,
            original_voxel_size=self.original_voxel_size,
            grid_shape=self.grid_shape,
            voxel_size=self.voxel_size,
            downsampling_factor=self.downsampling_factor,
            center_coordinate=self.center_coordinate,
            dtype=dtype,
            device=device,
        )


class TopLeftAlignedNormalizedFactory(IVoxelCoordinateSystemFactory):
    """Factory for creating top left aligned normalized coordinate systems"""

    def __init__(
        self,
        original_grid_shape: Sequence[int],
        original_voxel_size: Optional[Sequence[float]] = None,
        grid_shape: Optional[Sequence[int]] = None,
        voxel_size: Optional[Sequence[float]] = None,
        downsampling_factor: Optional[Sequence[float]] = None,
    ):
        self.original_grid_shape = original_grid_shape
        self.original_voxel_size = original_voxel_size
        self.grid_shape = grid_shape
        self.voxel_size = voxel_size
        self.downsampling_factor = downsampling_factor

    def create(self, dtype: torch_dtype, device: torch_device) -> VoxelCoordinateSystem:
        return create_top_left_aligned_normalized(
            original_grid_shape=self.original_grid_shape,
            original_voxel_size=self.original_voxel_size,
            grid_shape=self.grid_shape,
            voxel_size=self.voxel_size,
            downsampling_factor=self.downsampling_factor,
            dtype=dtype,
            device=device,
        )


class CenteredFactory(IVoxelCoordinateSystemFactory):
    """Factory for creating centered coordinate systems"""

    def __init__(
        self,
        original_grid_shape: Sequence[int],
        original_voxel_size: Optional[Sequence[float]] = None,
        grid_shape: Optional[Sequence[int]] = None,
        voxel_size: Optional[Sequence[float]] = None,
        downsampling_factor: Optional[Sequence[float]] = None,
        center_coordinate: Optional[Sequence[float]] = None,
    ):
        self.original_grid_shape = original_grid_shape
        self.original_voxel_size = original_voxel_size
        self.grid_shape = grid_shape
        self.voxel_size = voxel_size
        self.downsampling_factor = downsampling_factor
        self.center_coordinate = center_coordinate

    def create(self, dtype: torch_dtype, device: torch_device) -> VoxelCoordinateSystem:
        return create_centered(
            original_grid_shape=self.original_grid_shape,
            original_voxel_size=self.original_voxel_size,
            grid_shape=self.grid_shape,
            voxel_size=self.voxel_size,
            downsampling_factor=self.downsampling_factor,
            center_coordinate=self.center_coordinate,
            dtype=dtype,
            device=device,
        )


class TopLeftAlignedFactory(IVoxelCoordinateSystemFactory):
    """Factory for creating top left aligned coordinate systems"""

    def __init__(
        self,
        original_grid_shape: Sequence[int],
        original_voxel_size: Optional[Sequence[float]] = None,
        grid_shape: Optional[Sequence[int]] = None,
        voxel_size: Optional[Sequence[float]] = None,
        downsampling_factor: Optional[Sequence[float]] = None,
    ):
        self.original_grid_shape = original_grid_shape
        self.original_voxel_size = original_voxel_size
        self.grid_shape = grid_shape
        self.voxel_size = voxel_size
        self.downsampling_factor = downsampling_factor

    def create(self, dtype: torch_dtype, device: torch_device) -> VoxelCoordinateSystem:
        return create_top_left_aligned(
            original_grid_shape=self.original_grid_shape,
            original_voxel_size=self.original_voxel_size,
            grid_shape=self.grid_shape,
            voxel_size=self.voxel_size,
            downsampling_factor=self.downsampling_factor,
            dtype=dtype,
            device=device,
        )


class VoxelFactory(IVoxelCoordinateSystemFactory):
    """Factory for creating voxel coordinate systems"""

    def __init__(
        self,
        grid_shape: Sequence[int],
        voxel_size: Optional[Sequence[float]] = None,
    ):
        self.grid_shape = grid_shape
        self.voxel_size = voxel_size

    def create(self, dtype: torch_dtype, device: torch_device) -> VoxelCoordinateSystem:
        return create_voxel(
            grid_shape=self.grid_shape,
            voxel_size=self.voxel_size,
            dtype=dtype,
            device=device,
        )


def create_centered_normalized(
    original_grid_shape: Sequence[int],
    original_voxel_size: Optional[Sequence[float]] = None,
    grid_shape: Optional[Sequence[int]] = None,
    voxel_size: Optional[Sequence[float]] = None,
    downsampling_factor: Optional[Sequence[float]] = None,
    center_coordinate: Optional[Sequence[float]] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> VoxelCoordinateSystem:
    """Create centered and normalized coordinate system

    Origin is in the middle of the voxel space and voxels are assumed to be
    squares with the sampled value in the middle.

    Coordinates are scaled such that the whole FOV fits inside values from -1 to
    1 for the hypothetical original grid. The actual grid then has voxel size of
    the original grid multiplied by the downsampling factor and is located in
    the middle with respect to the the original grid.

    Args:
        original_grid_shape: Shape of the hypothetical original grid
        original_voxel_size: Voxel size of the hypothetical original grid
        grid_shape: Shape of the actual grid, defaults to original_grid_shape
        voxel_size: Voxel size of the actual grid, assumed to equal the
            original_voxel size if not given.
        downsampling_factor: Downsampling factor of the actual grid compared
            to the original grid, defaults to no scaling.
        center_coordinate: Center coordinate of the actual grid with respect to
            the hypothetical original grid
        dtype: Data type of the coordinate system
        device: Device of th coordinate system
    """
    original_voxel_size, voxel_size, grid_shape = _handle_optional_inputs(
        original_grid_shape=original_grid_shape,
        original_voxel_size=original_voxel_size,
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        downsampling_factor=downsampling_factor,
    )
    center_coordinate = (
        (0,) * len(original_grid_shape) if center_coordinate is None else center_coordinate
    )
    world_to_voxel_scale = _normalized_scale(
        original_grid_shape=original_grid_shape,
        original_voxel_size=original_voxel_size,
        voxel_size=voxel_size,
    )
    world_origin_in_voxels = [
        (dim_size - 1) / 2 - dim_center_coordinate * dim_world_to_voxel_scale
        for dim_size, dim_center_coordinate, dim_world_to_voxel_scale in zip(
            grid_shape, center_coordinate, world_to_voxel_scale
        )
    ]
    return _generate_coordinate_system(
        grid_shape=grid_shape,
        world_to_voxel_scale=world_to_voxel_scale,
        world_origin_in_voxels=world_origin_in_voxels,
        dtype=dtype,
        device=device,
    )


def create_top_left_aligned_normalized(
    original_grid_shape: Sequence[int],
    original_voxel_size: Optional[Sequence[float]] = None,
    grid_shape: Optional[Sequence[int]] = None,
    voxel_size: Optional[Sequence[float]] = None,
    downsampling_factor: Optional[Sequence[float]] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> VoxelCoordinateSystem:
    """Create top left aligned normalized coordinate system

    Voxels are assumed to be squares with the sampled value in the middle.

    Coordinates are scaled such that the whole FOV fits inside values from -1 to
    1 for the hypothetical original grid. The actual grid then has voxel size of
    the original grid multiplied by the downsampling factor and top-left corner
    is aligned with the original grid.

    Args:
        original_grid_shape: Shape of the hypothetical original grid
        original_voxel_size: Voxel size of the hypothetical original grid
        grid_shape: Shape of the actual grid, defaults to original_grid_shape
        voxel_size: Voxel size of the actual grid, assumed to equal the
            original_voxel size if not given.
        downsampling_factor: Downsampling factor of the actual grid compared
            to the original grid, defaults to no scaling.
        dtype: Data type of the coordinate system
        device: Device of th coordinate system
    """
    original_voxel_size, voxel_size, grid_shape = _handle_optional_inputs(
        original_grid_shape=original_grid_shape,
        original_voxel_size=original_voxel_size,
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        downsampling_factor=downsampling_factor,
    )
    world_to_voxel_scale = _normalized_scale(
        original_grid_shape=original_grid_shape,
        original_voxel_size=original_voxel_size,
        voxel_size=voxel_size,
    )
    world_origin_in_voxels = [
        (original_dim_size * original_dim_voxel_size / dim_voxel_size - 1) / 2
        for (original_dim_size, original_dim_voxel_size, dim_voxel_size) in zip(
            original_grid_shape, original_voxel_size, voxel_size
        )
    ]
    return _generate_coordinate_system(
        grid_shape=grid_shape,
        world_to_voxel_scale=world_to_voxel_scale,
        world_origin_in_voxels=world_origin_in_voxels,
        dtype=dtype,
        device=device,
    )


def create_centered(
    original_grid_shape: Sequence[int],
    original_voxel_size: Optional[Sequence[float]] = None,
    grid_shape: Optional[Sequence[int]] = None,
    voxel_size: Optional[Sequence[float]] = None,
    downsampling_factor: Optional[Sequence[float]] = None,
    center_coordinate: Optional[Sequence[float]] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> VoxelCoordinateSystem:
    """Create normalized coordinate system

    Origin is in the middle of the voxel space.

    The actual grid then has voxel size of the original grid multiplied by
    the downsampling factor.

    Args:
        original_grid_shape: Shape of the hypothetical original grid
        original_voxel_size: Voxel size of the hypothetical original grid
        grid_shape: Shape of the actual grid, defaults to original_grid_shape
        voxel_size: Voxel size of the actual grid, assumed to equal the
            original_voxel size if not given.
        downsampling_factor: Downsampling factor of the actual grid compared
            to the original grid, defaults to no scaling.
        center_coordinate: Center coordinate of the actual grid with respect to
            the hypothetical original grid
        dtype: Data type of the coordinate system
        device: Device of th coordinate system
    """
    original_voxel_size, voxel_size, grid_shape = _handle_optional_inputs(
        original_grid_shape=original_grid_shape,
        original_voxel_size=original_voxel_size,
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        downsampling_factor=downsampling_factor,
    )
    center_coordinate = (
        (0,) * len(original_grid_shape) if center_coordinate is None else center_coordinate
    )
    world_to_voxel_scale = [1 / dim_voxel_size for dim_voxel_size in voxel_size]
    world_origin_in_voxels = [
        (dim_size - 1) / 2 - dim_center_coordinate * dim_world_to_voxel_scale
        for dim_size, dim_center_coordinate, dim_world_to_voxel_scale in zip(
            grid_shape, center_coordinate, world_to_voxel_scale
        )
    ]
    return _generate_coordinate_system(
        grid_shape=grid_shape,
        world_to_voxel_scale=world_to_voxel_scale,
        world_origin_in_voxels=world_origin_in_voxels,
        dtype=dtype,
        device=device,
    )


def create_top_left_aligned(
    original_grid_shape: Sequence[int],
    original_voxel_size: Optional[Sequence[float]] = None,
    grid_shape: Optional[Sequence[int]] = None,
    voxel_size: Optional[Sequence[float]] = None,
    downsampling_factor: Optional[Sequence[float]] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> VoxelCoordinateSystem:
    """Create top left aligned coordinate system

    Voxels are assumed to be squares with the sampled value in the middle.

    The actual grid then has voxel size of the original grid multiplied by
    the downsampling factor and top-left corner is aligned with the original
    grid.

    Args:
        original_grid_shape: Shape of the hypothetical original grid
        original_voxel_size: Voxel size of the hypothetical original grid
        grid_shape: Shape of the actual grid, defaults to original_grid_shape
        voxel_size: Voxel size of the actual grid, assumed to equal the
            original_voxel size if not given.
        downsampling_factor: Downsampling factor of the actual grid compared
            to the original grid, defaults to no scaling.
        dtype: Data type of the coordinate system
        device: Device of th coordinate system
    """
    original_voxel_size, voxel_size, grid_shape = _handle_optional_inputs(
        original_grid_shape=original_grid_shape,
        original_voxel_size=original_voxel_size,
        grid_shape=grid_shape,
        voxel_size=voxel_size,
        downsampling_factor=downsampling_factor,
    )
    world_to_voxel_scale = [1 / dim_voxel_size for dim_voxel_size in voxel_size]
    n_dims = len(original_grid_shape)
    world_origin_in_voxels = [-1 / 2] * n_dims
    return _generate_coordinate_system(
        grid_shape=grid_shape,
        world_to_voxel_scale=world_to_voxel_scale,
        world_origin_in_voxels=world_origin_in_voxels,
        dtype=dtype,
        device=device,
    )


def create_voxel(
    grid_shape: Sequence[int],
    voxel_size: Optional[Sequence[float]] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[torch_device] = None,
) -> VoxelCoordinateSystem:
    """Create voxel coordinate system

    Args
        grid_shape: Shape of the grid
        voxel_size: Voxel size of the grid
        dtype: Data type of the coordinate system
        device: Device of th coordinate system
    """
    if voxel_size is None:
        voxel_size = [1.0] * len(grid_shape)
    return _generate_coordinate_system(
        grid_shape=grid_shape,
        world_to_voxel_scale=[1 / dim_voxel_size for dim_voxel_size in voxel_size],
        world_origin_in_voxels=[0.0] * len(grid_shape),
        dtype=dtype,
        device=device,
    )


def _handle_optional_inputs(
    original_grid_shape: Sequence[int],
    original_voxel_size: Optional[Sequence[float]],
    grid_shape: Optional[Sequence[int]],
    voxel_size: Optional[Sequence[float]],
    downsampling_factor: Optional[Sequence[float]],
) -> Tuple[Sequence[float], Sequence[float], Sequence[int]]:
    n_dims = len(original_grid_shape)
    if original_voxel_size is None:
        if voxel_size is not None:
            raise ValueError(
                "Providing only voxel_size is not allowed to avoid mistakes. "
                "Please provide also original_voxel_size."
            )
        original_voxel_size = [1.0] * n_dims
    if grid_shape is None:
        grid_shape = original_grid_shape
    if downsampling_factor is not None:
        inferred_voxel_size = [
            dim_original_voxel_size * dim_downsampling_factor
            for dim_original_voxel_size, dim_downsampling_factor in zip(
                original_voxel_size, downsampling_factor
            )
        ]
        if voxel_size is not None:
            if inferred_voxel_size != voxel_size:
                raise ValueError(
                    "Voxel size given and the one inferred using the downsampling factor "
                    "do not match. It is good idea to provide only either of them as providing"
                    "both of them is redundant."
                )
        voxel_size = inferred_voxel_size
    elif downsampling_factor is None and voxel_size is None:
        voxel_size = original_voxel_size
    elif downsampling_factor is not None and voxel_size is not None:
        raise ValueError("Provide only voxel size or downsampling factor")
    return original_voxel_size, voxel_size, grid_shape


def _generate_coordinate_system(
    grid_shape: Sequence[int],
    world_to_voxel_scale: Sequence[float],
    world_origin_in_voxels: Sequence[float],
    dtype: Optional[torch_dtype],
    device: Optional[torch_device],
) -> VoxelCoordinateSystem:
    voxel_to_world_scale = [1 / dim_voxel_size for dim_voxel_size in world_to_voxel_scale]
    transformation_matrix = _generate_scale_and_translation_matrix(
        scale=world_to_voxel_scale, translation=world_origin_in_voxels, dtype=dtype
    )
    to_voxel_coordinates_affine = CPUAffineTransformation(
        transformation_matrix, device=device, pin_memory=True
    )
    from_voxel_coordinates_affine = to_voxel_coordinates_affine.invert().reduce(pin_memory=True)
    to_voxel_coordinates = ComposableAffine(to_voxel_coordinates_affine)
    from_voxel_coordinates = ComposableAffine(from_voxel_coordinates_affine)
    voxel_grid = VoxelCoordinateGrid(grid_shape)
    return VoxelCoordinateSystem(
        from_voxel_coordinates=from_voxel_coordinates,
        to_voxel_coordinates=to_voxel_coordinates,
        grid=from_voxel_coordinates(voxel_grid),
        voxel_grid=voxel_grid,
        grid_spacing=voxel_to_world_scale,
    )


def _normalized_scale(
    original_grid_shape: Sequence[int],
    original_voxel_size: Sequence[float],
    voxel_size: Sequence[float],
) -> Sequence[float]:
    """Scaling applied to normalized coordinate systems

    Scaling is computed such that the whole volume just fits inside
    values [-1, 1].
    """
    scale = (
        max(
            original_dim_voxel_size * original_dim_size
            for (original_dim_voxel_size, original_dim_size) in zip(
                original_voxel_size, original_grid_shape
            )
        )
        / 2
    )
    world_to_voxel_scale = [scale / dim_voxel_size for dim_voxel_size in voxel_size]
    return world_to_voxel_scale


def _generate_scale_and_translation_matrix(
    scale: Sequence[float], translation: Sequence[float], dtype: Optional[torch_dtype]
) -> Tensor:
    matrix = diag(tensor(list(scale) + [1.0], dtype=dtype))
    matrix[:-1, -1] = tensor(translation, dtype=dtype)
    return matrix
