"""Voxel coordinate system"""

from dataclasses import dataclass
from typing import Mapping, Sequence

from torch import Tensor

from composable_mapping.interface import ITensorLike

from .base import BaseTensorLikeWrapper
from .interface import IComposableMapping, IMaskedTensor, IVoxelCoordinateSystem


@dataclass
class VoxelCoordinateSystem(IVoxelCoordinateSystem, BaseTensorLikeWrapper):
    """Represents coordinate system between voxel and world coordinates"""

    def __init__(
        self,
        from_voxel_coordinates: IComposableMapping,
        to_voxel_coordinates: IComposableMapping,
        grid: IMaskedTensor,
        voxel_grid: IMaskedTensor,
        grid_spacing: Sequence[float],
    ):
        self._from_voxel_coordinates = from_voxel_coordinates
        self._to_voxel_coordinates = to_voxel_coordinates
        self._grid = grid
        self._voxel_grid = voxel_grid
        self._grid_spacing = grid_spacing

    def _get_children(self) -> Mapping[str, ITensorLike]:
        return {
            "grid": self.grid,
            "voxel_grid": self.voxel_grid,
            "from_voxel_coordinates": self.from_voxel_coordinates,
            "to_voxel_coordinates": self.to_voxel_coordinates,
        }

    def _get_tensors(self) -> Mapping[str, Tensor]:
        return {}

    def _modified_copy(
        self, tensors: Mapping[str, Tensor], children: Mapping[str, ITensorLike]
    ) -> "VoxelCoordinateSystem":
        if (
            not isinstance(children["grid"], IMaskedTensor)
            or not isinstance(children["voxel_grid"], IMaskedTensor)
            or not isinstance(children["from_voxel_coordinates"], IComposableMapping)
            or not isinstance(children["to_voxel_coordinates"], IComposableMapping)
        ):
            raise ValueError("Invalid children for voxel coordinate system")
        return VoxelCoordinateSystem(
            from_voxel_coordinates=children["from_voxel_coordinates"],
            to_voxel_coordinates=children["to_voxel_coordinates"],
            grid=children["grid"],
            voxel_grid=children["voxel_grid"],
            grid_spacing=self.grid_spacing,
        )

    @property
    def from_voxel_coordinates(self) -> IComposableMapping:
        return self._from_voxel_coordinates

    @property
    def to_voxel_coordinates(self) -> IComposableMapping:
        return self._to_voxel_coordinates

    @property
    def grid(self) -> IMaskedTensor:
        return self._grid

    @property
    def voxel_grid(self) -> IMaskedTensor:
        return self._voxel_grid

    @property
    def grid_spacing(self) -> Sequence[float]:
        return self._grid_spacing

    def __repr__(self) -> str:
        return (
            "VoxelCoordinateSystem("
            f"from_voxel_coordinates={self.from_voxel_coordinates}, "
            f"to_voxel_coordinates={self.to_voxel_coordinates}, "
            f"grid={self.grid}, "
            f"voxel_grid={self.voxel_grid}, "
            f"grid_spacing={self.grid_spacing})"
        )
