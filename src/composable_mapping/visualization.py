"""Visualization utilities for composable mappings."""

from itertools import combinations
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from matplotlib.colors import Normalize, to_rgba  # type: ignore
from matplotlib.figure import Figure  # type: ignore
from matplotlib.pyplot import subplots  # type: ignore
from numpy import amax, amin, array, moveaxis, ndarray
from torch import Tensor
from torch import device as torch_device
from torch import ones, tensor

from .composable_mapping import (
    ComposableMapping,
    GridComposableMapping,
    ICoordinateSystemContainer,
)
from .interface import Number
from .mappable_tensor import MappableTensor
from .sampler import DataFormat
from .util import get_spatial_dims, get_spatial_shape


def obtain_coordinate_mapping_central_planes(
    volume: MappableTensor, batch_index: int = 0
) -> Tuple[Sequence[ndarray], Sequence[Tuple[int, int]]]:
    """Obtain central slices of a coordinate mapping along each channel
    dimension"""
    values = volume.generate_values()
    spatial_dims = get_spatial_dims(values.ndim, volume.n_channel_dims)
    n_dims = len(spatial_dims)
    values = values[batch_index]
    if n_dims > 1:
        slices = []
        dimension_pairs = list(combinations(range(n_dims), 2))
        for dimension_pair in dimension_pairs:
            other_dims = [dim for dim in range(n_dims) if dim not in dimension_pair]
            transformed_grid_2d = values[list(dimension_pair)]
            for other_dim in reversed(other_dims):
                transformed_grid_2d = transformed_grid_2d.movedim(-n_dims + other_dim, 0)
                transformed_grid_2d = transformed_grid_2d[transformed_grid_2d.size(0) // 2]
            slices.append(_to_numpy(transformed_grid_2d))
    else:
        raise NotImplementedError("Currently 1D volumes are not supported")
    return slices, dimension_pairs


def obtain_central_planes(
    volume: MappableTensor,
) -> Tuple[Sequence[ndarray], Sequence[ndarray], Sequence[Tuple[int, int]]]:
    """Obtain central slices of a volume along each channel dimension"""
    values, mask = volume.generate()
    spatial_dims = get_spatial_dims(values.ndim, volume.n_channel_dims)
    n_dims = len(spatial_dims)
    if n_dims > 1:
        planes = []
        mask_planes = []
        dimension_pairs = list(combinations(range(n_dims), 2))
        for dimension_pair in dimension_pairs:
            other_dims = [dim for dim in range(n_dims) if dim not in dimension_pair]
            plane = values
            mask_plane = mask
            for other_dim in reversed(other_dims):
                plane = plane.movedim(spatial_dims[other_dim], 1)
                plane = plane[:, plane.size(1) // 2]
                mask_plane = mask_plane.movedim(spatial_dims[other_dim], 1)
                mask_plane = mask_plane[:, mask_plane.size(0) // 2]
            spatial_shape = get_spatial_shape(plane.shape, n_channel_dims=1)
            if spatial_shape[0] != 1 and spatial_shape[1] != 1:
                planes.append(_to_numpy(plane))
                mask_planes.append(_to_numpy(mask_plane))
        return planes, mask_planes, dimension_pairs
    raise NotImplementedError("Currently 1D volumes are not supported")


def dimension_to_letter(dim: int, n_dims: int) -> str:
    """Convert dimension index to letter"""
    if n_dims <= 3:
        return "xyz"[dim]
    return f"dim_{dim}"


class GridVisualizationArguments:
    """Arguments for grid visualization"""

    def __init__(
        self,
        batch_index: int = 0,
        figure_height: Number = 5,
        emphasize_every_nth_line: Optional[Tuple[int, int]] = (5, 2),
        plot_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.batch_index = batch_index
        self.figure_height = figure_height
        self.emphasize_every_nth_line = emphasize_every_nth_line
        self.plot_kwargs = {} if plot_kwargs is None else plot_kwargs


def visualize_grid(
    coordinates: MappableTensor,
    arguments: Optional[GridVisualizationArguments] = None,
) -> Figure:
    """Visualize coordinates as a grid"""
    if coordinates.n_channel_dims != 1:
        raise ValueError("Only single-channel coordinates are supported")
    if coordinates.channels_shape[0] != len(coordinates.spatial_shape):
        raise ValueError("Number of channels must match number of spatial dimensions")
    if arguments is None:
        arguments = GridVisualizationArguments()
    n_dims = len(coordinates.spatial_shape)
    planes, _mask_planes, dimension_pairs = obtain_central_planes(coordinates)
    if arguments.plot_kwargs is None:
        arguments.plot_kwargs = {}

    def get_kwargs(index: int) -> Mapping[str, Any]:
        if arguments.emphasize_every_nth_line is None:
            return arguments.plot_kwargs
        if (index + arguments.emphasize_every_nth_line[1]) % arguments.emphasize_every_nth_line[
            0
        ] == 0:
            kwargs = {"alpha": 0.6, "linewidth": 2.0}
        else:
            kwargs = {"alpha": 0.2, "linewidth": 1.0}
        kwargs.update(arguments.plot_kwargs)
        return kwargs

    figure, axes = subplots(
        1,
        len(planes),
        figsize=(arguments.figure_height * len(planes), arguments.figure_height),
        squeeze=False,
    )

    for axis, plane, (dim_1, dim_2) in zip(axes.flatten(), planes, dimension_pairs):
        plane = plane[arguments.batch_index, [dim_1, dim_2]]
        axis.axis("equal")
        axis.set_xlabel(dimension_to_letter(dim_1, n_dims))
        axis.set_ylabel(dimension_to_letter(dim_2, n_dims))
        for row_index in range(plane.shape[1]):
            axis.plot(
                plane[0, row_index, :],
                plane[1, row_index, :],
                color="gray",
                **get_kwargs(row_index),
            )
        for col_index in range(plane.shape[2]):
            axis.plot(
                plane[0, :, col_index],
                plane[1, :, col_index],
                color="gray",
                **get_kwargs(col_index),
            )

    return figure


class ImageVisualizationArguments:
    """Arguments for image visualization"""

    def __init__(
        self,
        batch_index: int = 0,
        figure_height: Number = 5,
        mask_color: Optional[Any] = "red",
        mask_alpha: Number = 0.5,
        vmin: Optional[Number] = None,
        vmax: Optional[Number] = None,
        imshow_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.batch_index = batch_index
        self.figure_height = figure_height
        self.mask_color = mask_color
        self.mask_alpha = mask_alpha
        self.vmin = vmin
        self.vmax = vmax
        self.imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs


def visualize_image(
    volume: MappableTensor,
    voxel_size: Optional[Union[Tensor, Number, Sequence[Number]]] = None,
    arguments: Optional[ImageVisualizationArguments] = None,
) -> Figure:
    """Visualize coordinates as an image"""
    if volume.n_channel_dims != 1:
        raise ValueError("Only single-channel volumes are supported")
    if arguments is None:
        arguments = ImageVisualizationArguments()
    if voxel_size is None:
        voxel_size = ones(len(volume.spatial_shape), device=torch_device("cpu"))
    if not isinstance(voxel_size, Tensor):
        voxel_size = tensor(voxel_size, device=torch_device("cpu"))
    voxel_size = voxel_size.expand(volume.batch_shape[0], len(volume.spatial_shape))
    if volume.channels_shape[0] == 1:
        cmap: Optional[Any] = "gray"
    else:
        cmap = None
    n_dims = len(volume.spatial_shape)
    planes, mask_planes, dimension_pairs = obtain_central_planes(volume)
    vmin = (
        min(amin(plane[arguments.batch_index]) for plane in planes)
        if arguments.vmin is None
        else arguments.vmin
    )
    vmax = (
        max(amax(plane[arguments.batch_index]) for plane in planes)
        if arguments.vmax is None
        else arguments.vmax
    )
    normalizer = Normalize(vmin=vmin, vmax=vmax)
    figure, axes = subplots(
        1,
        len(planes),
        figsize=(arguments.figure_height * len(planes), arguments.figure_height),
        squeeze=False,
    )
    if arguments.mask_color is None:
        mask_color: Optional[ndarray] = None
    else:
        mask_color = array(to_rgba(arguments.mask_color))[None, None]
        mask_color[..., -1] = arguments.mask_alpha
    for axis, plane, mask_plane, (dim_1, dim_2) in zip(
        axes.flatten(), planes, mask_planes, dimension_pairs
    ):
        plane = normalizer(plane)
        aspect = (
            voxel_size[arguments.batch_index, dim_1].item()
            / voxel_size[arguments.batch_index, dim_2].item()
        )
        plane = moveaxis(plane[arguments.batch_index], 0, -1)
        axis.set_ylabel(dimension_to_letter(dim_1, n_dims))
        axis.set_xlabel(dimension_to_letter(dim_2, n_dims))

        axis.imshow(
            plane,
            origin="lower",
            aspect=aspect,
            cmap=cmap,
            **arguments.imshow_kwargs,
        )
        if mask_color is not None:
            mask_plane = moveaxis(mask_plane[arguments.batch_index], 0, -1)
            coloured_mask_plane = (1 - mask_plane) * mask_color
            axis.imshow(
                coloured_mask_plane,
                origin="lower",
                aspect=aspect,
                **arguments.imshow_kwargs,
            )

    return figure


def visualize_as_image(
    mapping: GridComposableMapping,
    arguments: Optional[ImageVisualizationArguments] = None,
):
    """Visualize a grid mapping as an image"""
    return visualize_image(
        mapping.sample(),
        voxel_size=mapping.coordinate_system.grid_spacing_cpu(),
        arguments=arguments,
    )


def visualize_to_as_image(
    mapping: ComposableMapping,
    target: ICoordinateSystemContainer,
    arguments: Optional[ImageVisualizationArguments] = None,
):
    """Visualize a mapping to a target coordinate system as an image"""
    return visualize_image(
        mapping.sample_to(target),
        voxel_size=target.coordinate_system.grid_spacing_cpu(),
        arguments=arguments,
    )


def visualize_as_deformed_grid(
    mapping: GridComposableMapping,
    arguments: Optional[GridVisualizationArguments] = None,
):
    """Visualize a grid mapping as a grid"""
    return visualize_to_as_deformed_grid(mapping, target=mapping, arguments=arguments)


def visualize_to_as_deformed_grid(
    mapping: ComposableMapping,
    target: ICoordinateSystemContainer,
    arguments: Optional[GridVisualizationArguments] = None,
):
    """Visualize a mapping to a target coordinate system as a grid"""
    return visualize_grid(
        mapping.sample_to(target, data_format=DataFormat.world_coordinates()), arguments=arguments
    )


def _to_numpy(item: Tensor) -> ndarray:
    return item.detach().cpu().resolve_conj().resolve_neg().numpy()
