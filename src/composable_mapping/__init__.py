"""Library for handling spatial coordinate transformations in PyTorch.

Developed originally for medical imaging, this library provides a set of classes
and functions for handling spatial coordinate transformations.

The most powerful feature of this library is the ability to easily compose
transformations lazily and resample them to different coordinate systems as well
as sampler classes for sampling volumes defined on regular grids such that the
optimal method (either convolution or torch.grid_sample) is used based on the
sampling locations.

The main idea was to develop a library that allows handling of the coordinate
mappings as if they were mathematical functions, without losing much performance
compared to more manual implementation.
"""

from .affine_mapping import Affine, affine, diagonal_affine
from .composable_mapping import (
    ComposableMapping,
    GridComposableMapping,
    Identity,
    SamplableVolume,
    concatenate_mappings,
    samplable_volume,
    stack_mappings,
)
from .coordinate_system import (
    Center,
    CoordinateSystem,
    End,
    OriginalFOV,
    OriginalShape,
    ReformattingReference,
    ReformattingSpatialShape,
    Start,
)
from .derivative import estimate_spatial_jacobian_matrices
from .interface import Number
from .mappable_tensor import (
    MappableTensor,
    concatenate_mappable_tensors,
    mappable,
    stack_mappable_tensors,
    voxel_grid,
)
from .mask import ClearMask, RectangleMask
from .sampler import (
    BaseSeparableSampler,
    BicubicInterpolator,
    CubicSplineSampler,
    DataFormat,
    GenericSeparableDerivativeSampler,
    ISampler,
    LimitDirection,
    LinearInterpolator,
    NearestInterpolator,
    ScalingAndSquaring,
    clear_default_sampler,
    default_sampler,
    get_default_sampler,
    get_sampler,
    set_default_sampler,
)
from .tensor_like import ITensorLike
from .visualization import (
    GridVisualizationArguments,
    ImageVisualizationArguments,
    visualize_as_deformed_grid,
    visualize_as_image,
    visualize_grid,
    visualize_image,
    visualize_to_as_deformed_grid,
    visualize_to_as_image,
)

__all__ = [
    "Affine",
    "CoordinateSystem",
    "ReformattingReference",
    "Center",
    "End",
    "Start",
    "OriginalShape",
    "OriginalFOV",
    "ReformattingSpatialShape",
    "ComposableMapping",
    "GridComposableMapping",
    "Identity",
    "SamplableVolume",
    "concatenate_mappings",
    "samplable_volume",
    "stack_mappings",
    "estimate_spatial_jacobian_matrices",
    "Number",
    "MappableTensor",
    "concatenate_mappable_tensors",
    "mappable",
    "stack_mappable_tensors",
    "voxel_grid",
    "ClearMask",
    "RectangleMask",
    "BaseSeparableSampler",
    "BicubicInterpolator",
    "CubicSplineSampler",
    "DataFormat",
    "GenericSeparableDerivativeSampler",
    "ISampler",
    "LimitDirection",
    "LinearInterpolator",
    "NearestInterpolator",
    "ScalingAndSquaring",
    "clear_default_sampler",
    "default_sampler",
    "get_default_sampler",
    "get_sampler",
    "set_default_sampler",
    "ITensorLike",
    "GridVisualizationArguments",
    "ImageVisualizationArguments",
    "visualize_as_deformed_grid",
    "visualize_as_image",
    "visualize_grid",
    "visualize_image",
    "visualize_to_as_deformed_grid",
    "visualize_to_as_image",
]
