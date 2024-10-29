from itertools import combinations
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from matplotlib.figure import Figure  # type: ignore
from matplotlib.pyplot import subplots  # type: ignore
from numpy import moveaxis, ndarray
from torch import Tensor

from .mappable_tensor import MappableTensor, mappable
from .util import get_spatial_dims, to_numpy

Number = Union[float, int]


def obtain_central_planes(
    volume: MappableTensor, batch_index: int = 0
) -> Tuple[Sequence[ndarray], Sequence[Tuple[int, int]]]:
    """Obtain central slices of a volume along each channel dimension"""
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
            slices.append(to_numpy(transformed_grid_2d))
    else:
        raise NotImplementedError("Currently 1D volumes are not supported")
    return slices, dimension_pairs


def dimension_to_letter(dim: int, n_dims: int) -> str:
    """Convert dimension index to letter"""
    if n_dims <= 3:
        return "xyz"[dim]
    return f"dim_{dim}"


def visualize_as_grid(
    coordinates: MappableTensor,
    batch_index: int = 0,
    figure_height: Number = 5,
    emphasize_every_nth_line: Optional[Tuple[int, int]] = None,
    plot_kwargs: Optional[Mapping[str, Any]] = None,
) -> Figure:
    """Visualize coordinates as a grid"""
    if coordinates.n_channel_dims != 1:
        raise ValueError("Only single-channel coordinates are supported")
    if coordinates.channels_shape[0] != len(coordinates.spatial_shape):
        raise ValueError("Number of channels must match number of spatial dimensions")
    n_dims = len(coordinates.spatial_shape)
    grids, dimension_pairs = obtain_central_planes(coordinates, batch_index=batch_index)
    if plot_kwargs is None:
        plot_kwargs = {}

    def get_kwargs(index: int) -> Mapping[str, Any]:
        if emphasize_every_nth_line is None:
            return plot_kwargs
        if (index + emphasize_every_nth_line[1]) % emphasize_every_nth_line[0] == 0:
            kwargs = {"alpha": 0.6, "linewidth": 2.0}
        else:
            kwargs = {"alpha": 0.2, "linewidth": 1.0}
        kwargs.update(plot_kwargs)
        return kwargs

    figure, axes = subplots(
        1, len(grids), figsize=(figure_height * len(grids), figure_height), squeeze=False
    )

    for axis, grid, (dim_1, dim_2) in zip(axes.flatten(), grids, dimension_pairs):
        axis.axis("equal")
        axis.set_xlabel(dimension_to_letter(dim_1, n_dims))
        axis.set_ylabel(dimension_to_letter(dim_2, n_dims))
        for row_index in range(grid.shape[1]):
            axis.plot(
                grid[0, row_index, :],
                grid[1, row_index, :],
                color="gray",
                **get_kwargs(row_index),
            )
        for col_index in range(grid.shape[2]):
            axis.plot(
                grid[0, :, col_index],
                grid[1, :, col_index],
                color="gray",
                **get_kwargs(col_index),
            )

    return figure


def visualize_as_image(
    volume: MappableTensor,
    voxel_size: Tensor,
    batch_index: int = 0,
    figure_height: Number = 5,
    multiply_by_mask: bool = False,
    imshow_kwargs: Optional[Mapping[str, Any]] = None,
) -> Figure:
    """Visualize coordinates as an image"""
    if volume.n_channel_dims != 1:
        raise ValueError("Only single-channel volumes are supported")
    n_dims = len(volume.spatial_shape)
    volume = volume.reduce()
    values = volume.generate_values()
    if multiply_by_mask:
        mask = volume.generate_mask(generate_missing_mask=True, cast_mask=True)
        values = values * mask
        volume = mappable(values, mask, n_channel_dims=volume.n_channel_dims)
    min_value = values.amin().item()
    max_value = values.amax().item()
    kwargs: Dict[str, Any] = {"vmin": min_value, "vmax": max_value}
    if volume.channels_shape[-1] == 1:
        kwargs["cmap"] = "gray"
    if imshow_kwargs is not None:
        kwargs.update(imshow_kwargs)
    grids, dimension_pairs = obtain_central_planes(volume, batch_index=batch_index)
    figure, axes = subplots(
        1, len(grids), figsize=(figure_height * len(grids), figure_height), squeeze=False
    )

    for axis, grid, (dim_1, dim_2) in zip(axes.flatten(), grids, dimension_pairs):
        aspect = voxel_size[dim_1].item() / voxel_size[dim_2].item()
        grid = moveaxis(grid, 0, -1)
        if grid.shape[-1] == 1:
            grid = grid[..., 0]
        axis.set_xlabel(dimension_to_letter(dim_1, n_dims))
        axis.set_ylabel(dimension_to_letter(dim_2, n_dims))
        axis.imshow(
            grid,
            origin="lower",
            aspect=aspect,
            **kwargs,
        )

    return figure
