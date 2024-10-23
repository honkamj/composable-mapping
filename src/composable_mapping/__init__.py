"""Composable mapping library"""

__all__ = ["coordinate_system", "mappable_tensor", "grid_mapping", "tensor_like"]

from .affine import Affine, as_affine_transformation
from .coordinate_system import CoordinateSystem
from .identity import Identity
from .interface import IComposableMapping, ISampler
from .mappable_tensor import MappableTensor, PlainTensor, VoxelGrid
from .mask import ClearMask, RectangleMask
from .samplable_mapping import GridDeformation, GridVolume
from .sampler import BicubicInterpolator, LinearInterpolator, NearestInterpolator
