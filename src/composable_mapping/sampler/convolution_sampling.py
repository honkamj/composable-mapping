"""Core functions for convolution sampling."""

from math import ceil, floor
from typing import List, Optional, Tuple

from torch import Tensor
from torch import device as torch_device
from torch import diag, empty, linspace, ones, tensor, unique, zeros
from torch.jit import script

from composable_mapping.util import get_spatial_dims


@script
def obtain_normalizing_flipping_permutation(
    matrix: Tensor,
) -> Optional[Tuple[List[int], List[int]]]:
    """Obtain permutation with flipping which turns the affine matrix as
    close to a diagonal matrix with positive diagonal as possible.

    If the algorithm is not able to find a valid permutation, None is returned.
    This could be improved as the current algorithm can not handle zero diagonal elements.

    Args:
        matrix: The affine matrix (without the last row) to normalize
            with shape (n_dims, n_dims + 1).

    Returns:
        Normalizing flipping permutation or None if no valid flipping
        permutation is found.
    """
    spatial_permutation = matrix[:, :-1].abs().argmax(dim=0)
    n_dims = matrix.shape[-1] - 1
    if len(unique(spatial_permutation)) != n_dims:
        return None
    spatial_permutation_list: List[int] = spatial_permutation.tolist()
    flipped_spatial_dims: List[int] = []
    for column, largest_row in enumerate(spatial_permutation_list):
        if matrix[largest_row, column] < 0:
            flipped_spatial_dims.append(largest_row)
    return spatial_permutation_list, flipped_spatial_dims


@script
def apply_flipping_permutation_to_affine_matrix(
    matrix: Tensor,
    spatial_permutation: List[int],
    flipped_spatial_dims: List[int],
    volume_spatial_shape: List[int],
) -> Tensor:
    """Apply the flipping and permutation to an affine matrix.

    Args:
        matrix: Affine matrix (without the last row) to transform
            with shape (batch_size, n_dims, n_dims + 1).
        spatial_shape: Shape of the spatial dimensions.

    Returns:
        Transformed affine matrix.
    """
    n_dims = len(volume_spatial_shape)
    if matrix.shape[-2:] != (n_dims, n_dims + 1):
        raise ValueError("Matrix has wrong shape")

    if flipped_spatial_dims:
        flip_vector = ones(n_dims, dtype=matrix.dtype, device=matrix.device)
        translation_vector = zeros(n_dims, dtype=matrix.dtype, device=matrix.device)
        for flipped_spatial_dim in flipped_spatial_dims:
            flip_vector[flipped_spatial_dim] = -1.0
            translation_vector[flipped_spatial_dim] = volume_spatial_shape[flipped_spatial_dim] - 1
        matrix = (matrix * flip_vector[:, None]).contiguous()
        matrix[:, :, -1] += translation_vector
    elif spatial_permutation == list(range(n_dims)):
        return matrix
    return matrix[:, spatial_permutation, :]


@script
def extract_conv_samplable_parameters(
    volume_spatial_shape: List[int],
    grid_spatial_shape: List[int],
    grid_affine_matrix: Tensor,
    is_interpolating_kernel: List[bool],
    convolution_threshold: float,
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, List[int], List[int]]]:
    """Extract parameters which can be used to sample the grid using
    convolutional interpolation.

    Args:
        volume_spatial_shape: Shape of the spatial dimensions of the volume.
        grid_spatial_shape: Shape of the spatial dimensions of the grid.
        grid_affine_matrix: Affine matrix applied to the voxel grid.
        is_interpolating_kernel: Whether the kernel is interpolating in each spatial dimension.
        convolution_threshold: Threshold for the maximum coordinate difference
            for using convolution.

    Returns:
        Tuple of downsampling factor, translation, convolve, transposed_convolve,
        spatial_permutation and flipped_spatial_dims or None if the sampling can not
        be done using convolution.
    """
    if grid_affine_matrix.shape[-2] != grid_affine_matrix.shape[-1]:
        return None
    n_dims = len(grid_spatial_shape)
    grid_affine_matrix = grid_affine_matrix.view(-1, n_dims + 1, n_dims + 1)[:, :-1, :]
    flipping_permutation = obtain_normalizing_flipping_permutation(grid_affine_matrix[0])
    if flipping_permutation is None:
        return None
    spatial_permutation, flipped_spatial_dims = flipping_permutation
    grid_affine_matrix = apply_flipping_permutation_to_affine_matrix(
        grid_affine_matrix,
        spatial_permutation=spatial_permutation,
        flipped_spatial_dims=flipped_spatial_dims,
        volume_spatial_shape=volume_spatial_shape,
    )
    diagonal = grid_affine_matrix[0, :, :-1].diagonal()
    if diagonal.any() == 0.0:
        return None
    transposed_convolve = diagonal < 0.75
    device = grid_affine_matrix.device
    downsampling_factor = empty(diagonal.shape, device=device, dtype=diagonal.dtype)
    downsampling_factor[~transposed_convolve] = diagonal[~transposed_convolve].round()
    downsampling_factor[transposed_convolve] = 1 / (1 / diagonal[transposed_convolve]).round()
    rounded_diagonal_matrix = diag(downsampling_factor)
    difference_matrix = grid_affine_matrix[:, :, :-1] - rounded_diagonal_matrix
    shape_tensor = tensor(grid_spatial_shape, device=device, dtype=grid_affine_matrix.dtype)
    max_diagonal_coordinate_difference_upper_bound = (
        (difference_matrix * shape_tensor).abs().amax(dim=(0, 2))
    )
    if grid_affine_matrix.size(0) == 1:
        max_translation_coordinate_difference: Tensor = tensor(
            0.0, device=device, dtype=diagonal.dtype
        )
    else:
        max_translation_coordinate_difference = (
            (grid_affine_matrix[1:, :, -1] - grid_affine_matrix[0, :, -1]).abs().amax(dim=0)
        )
    max_conv_coordinate_difference_upper_bound = (
        max_diagonal_coordinate_difference_upper_bound + max_translation_coordinate_difference
    )
    if (max_conv_coordinate_difference_upper_bound > convolution_threshold).any():
        return None
    interpolating_kernel = tensor(
        [
            is_interpolating_kernel[spatial_permutation[spatial_dim]]
            for spatial_dim in range(n_dims)
        ],
        device=device,
    )
    translation = grid_affine_matrix[0, :, -1].clone().contiguous()
    rounded_translation = translation.round()
    small_translation = (
        (translation - rounded_translation).abs() + max_conv_coordinate_difference_upper_bound
    ) < convolution_threshold
    if (interpolating_kernel & (~transposed_convolve)).any():
        convolve = ((~interpolating_kernel) | (~small_translation)) & (~transposed_convolve)
        no_convolve_or_transposed_convolve = (~transposed_convolve) & (~convolve)
        translation[no_convolve_or_transposed_convolve] = translation[
            no_convolve_or_transposed_convolve
        ].round()
    else:
        convolve = ~transposed_convolve
    return (
        downsampling_factor,
        translation,
        convolve,
        transposed_convolve,
        spatial_permutation,
        flipped_spatial_dims,
    )


def apply_flipping_permutation_to_volume(
    volume: Tensor,
    n_channel_dims: int,
    spatial_permutation: List[int],
    flipped_spatial_dims: List[int],
) -> Tensor:
    """Apply the flipping and permutation to a volume.

    Args:
        volume: Volume to transform with shape (*batch_shape, *channels_shape, *spatial_shape).
        spatial_permutation: Spatial permutation to apply.
        flipped_spatial_dims: Spatial dimensions to flip.
    """
    spatial_dims = get_spatial_dims(volume.ndim, n_channel_dims=n_channel_dims)
    if flipped_spatial_dims:
        flipped_dims = [spatial_dims[spatial_dim] for spatial_dim in flipped_spatial_dims]
        volume = volume.flip(dims=flipped_dims)
    permuted_spatial_dims = permute_sequence(spatial_dims, spatial_permutation)
    volume = volume.permute(tuple(range(spatial_dims[0])) + tuple(permuted_spatial_dims))
    return volume


@script
def permute_sequence(sequence: List[int], permutation: List[int]) -> List[int]:
    """Permute a sequence according to a permutation.

    Args:
        sequence: List to permute.
        permutation: Permutation to apply.

    Returns:
        Permuted sequence.
    """
    if len(sequence) != len(permutation):
        raise ValueError("Sequence has wrong length.")
    permuted: List[int] = []
    for permuted_dim in permutation:
        permuted.append(sequence[permuted_dim])
    return permuted


@script
def _optionally_inclusive_floor(
    value: float,
    inclusive: bool,
) -> int:
    if inclusive:
        return int(floor(value))
    return int(ceil(value - 1))


@script
def obtain_conv_parameters(
    volume_spatial_shape: List[int],
    grid_spatial_shape: List[int],
    grid_affine_matrix: Tensor,
    is_interpolating_kernel: List[bool],
    kernel_support: List[Tuple[float, float, bool, bool]],
    convolution_threshold: float,
    target_device: torch_device,
) -> Optional[
    Tuple[
        List[Optional[Tensor]],
        List[int],
        List[int],
        List[bool],
        List[Tuple[int, int]],
        List[Tuple[int, int]],
        List[int],
        List[int],
    ]
]:
    """Obtain parameters for convolutional sampling.

    Args:
        volume_spatial_shape: Shape of the spatial dimensions of the volume.
        grid_spatial_shape: Shape of the spatial dimensions of the grid.
        grid_affine_matrix: Affine matrix applied to the voxel grid.
        is_interpolating_kernel: Whether the kernel is interpolating in each spatial dimension.
        kernel_support: Support of the kernel in each spatial dimension.
        convolution_threshold: Threshold for the maximum coordinate difference
            for using convolution.
        target_device: Device to use for the kernel coordinates.

    Returns:
        Tuple of kernel coordinates, strides, paddings, transposed convolve, pre
        pads or crops, post pads or crops, spatial permutation and flipped
        spatial dimensions or None if the sampling can not be done using
        convolution.
    """
    conv_samplable_parameters = extract_conv_samplable_parameters(
        volume_spatial_shape=volume_spatial_shape,
        grid_spatial_shape=grid_spatial_shape,
        grid_affine_matrix=grid_affine_matrix,
        is_interpolating_kernel=is_interpolating_kernel,
        convolution_threshold=convolution_threshold,
    )
    if conv_samplable_parameters is None:
        return None
    (
        downsampling_factor,
        translation,
        convolve,
        transposed_convolve,
        spatial_permutation,
        flipped_spatial_dims,
    ) = conv_samplable_parameters
    downsampling_factor_list: List[float] = downsampling_factor.tolist()
    translation_list: List[float] = translation.tolist()
    convolve_list: List[bool] = convolve.tolist()
    transposed_convolve_list: List[bool] = transposed_convolve.tolist()
    pre_pads_or_crops: List[Tuple[int, int]] = []
    post_pads_or_crops: List[Tuple[int, int]] = []
    conv_paddings: List[int] = []
    conv_kernel_coordinates: List[Optional[Tensor]] = []
    for spatial_dim, (
        dim_size_volume,
        dim_size_grid,
        dim_convolve,
        dim_transposed_convolve,
        dim_translation,
        dim_downsampling_factor,
    ) in enumerate(
        zip(
            permute_sequence(volume_spatial_shape, spatial_permutation),
            grid_spatial_shape,
            convolve_list,
            transposed_convolve_list,
            translation_list,
            downsampling_factor_list,
        )
    ):
        kernel_dim = spatial_permutation[spatial_dim]
        flip_kernel = kernel_dim in flipped_spatial_dims
        (kernel_min, kernel_max, inclusive_min, inclusive_max) = kernel_support[kernel_dim]
        if flip_kernel:
            inclusive_min, inclusive_max = inclusive_max, inclusive_min
        min_coordinate = dim_translation
        max_coordinate = dim_translation + dim_downsampling_factor * (dim_size_grid - 1)
        if dim_convolve or dim_transposed_convolve:
            pre_pad_or_crop_lower = _optionally_inclusive_floor(
                -kernel_min - dim_translation, inclusive=inclusive_min
            )
            pre_pad_or_crop_upper = _optionally_inclusive_floor(
                kernel_max
                + dim_translation
                + dim_downsampling_factor * (dim_size_grid - 1)
                - (dim_size_volume - 1),
                inclusive=inclusive_max,
            )
            if dim_transposed_convolve:
                start_kernel_coordinate = (
                    1
                    - dim_translation
                    + _optionally_inclusive_floor(
                        (kernel_max - (1 - dim_translation)) / dim_downsampling_factor,
                        inclusive=inclusive_max,
                    )
                    * dim_downsampling_factor
                )
                end_kernel_coordinate = (
                    -dim_translation
                    - _optionally_inclusive_floor(
                        (-kernel_min - dim_translation) / dim_downsampling_factor,
                        inclusive=inclusive_min,
                    )
                    * dim_downsampling_factor
                )
                kernel_step_size = dim_downsampling_factor
            else:
                relative_coordinate = dim_translation - floor(dim_translation)
                start_kernel_coordinate = (
                    -_optionally_inclusive_floor(
                        -kernel_min - relative_coordinate, inclusive=inclusive_min
                    )
                    - relative_coordinate
                )
                end_kernel_coordinate = _optionally_inclusive_floor(
                    kernel_max - (1 - relative_coordinate), inclusive=inclusive_max
                ) + (1 - relative_coordinate)
                kernel_step_size = 1.0
            kernel_coordinates = linspace(
                start_kernel_coordinate,
                end_kernel_coordinate,
                int(round(abs(end_kernel_coordinate - start_kernel_coordinate) / kernel_step_size))
                + 1,
                dtype=grid_affine_matrix.dtype,
                device=target_device,
            )
            if flip_kernel:
                kernel_coordinates = -kernel_coordinates
        else:
            kernel_coordinates = None
            start_kernel_coordinate = 0.0  # dummy value for torchscript compatibility
            end_kernel_coordinate = 0.0  # dummy value for torchscript compatibility
            pre_pad_or_crop_lower = -int(min_coordinate)
            pre_pad_or_crop_upper = int(max_coordinate) - (dim_size_volume - 1)
        if dim_transposed_convolve:
            post_pad_or_crop_lower = -int(
                round(
                    (min_coordinate + pre_pad_or_crop_lower + start_kernel_coordinate)
                    / dim_downsampling_factor
                )
            )
            post_pad_or_crop_upper = -int(
                round(
                    (
                        (dim_size_volume - 1)
                        + pre_pad_or_crop_upper
                        - end_kernel_coordinate
                        - max_coordinate
                    )
                    / dim_downsampling_factor
                )
            )
            assert post_pad_or_crop_lower <= 0 and post_pad_or_crop_upper <= 0
            conv_padding = -max(post_pad_or_crop_lower, post_pad_or_crop_upper)
            post_pad_or_crop_lower += conv_padding
            post_pad_or_crop_upper += conv_padding
        else:
            conv_padding = 0
            post_pad_or_crop_lower = 0
            post_pad_or_crop_upper = 0
        conv_kernel_coordinates.append(kernel_coordinates)
        conv_paddings.append(conv_padding)
        pre_pads_or_crops.append((pre_pad_or_crop_lower, pre_pad_or_crop_upper))
        post_pads_or_crops.append((post_pad_or_crop_lower, post_pad_or_crop_upper))
    conv_strides = (
        (
            downsampling_factor * (~transposed_convolve)
            + (1 / downsampling_factor) * transposed_convolve
        )
        .round()
        .long()
    )
    conv_strides_list: List[int] = conv_strides.tolist()
    return (
        conv_kernel_coordinates,
        conv_strides_list,
        conv_paddings,
        transposed_convolve_list,
        pre_pads_or_crops,
        post_pads_or_crops,
        spatial_permutation,
        flipped_spatial_dims,
    )
