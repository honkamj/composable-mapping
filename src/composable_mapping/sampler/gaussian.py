"""Gaussian smoothing sampler."""

from math import pi, sqrt
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from torch import Tensor
from torch import bool as torch_bool
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import exp, linspace, tensor, zeros

from composable_mapping.interface import Number

from .separable_sampler import PiecewiseKernelDefinition, SeparableSampler


class GaussianKernel(PiecewiseKernelDefinition):
    """Gaussian kernel.

    Arguments:
        truncate_at: Truncation of the kernel. Can be a single value, tuple of
            min and max, or a list of tuples of min and max for each spatial
            dimension.
        mean: Mean of the Gaussian kernel. Can be a single value, or a separate
            value for each spatial dimension.
        std: Standard deviation of the Gaussian kernel. Can be a single value,
            or a separate value for each spatial dimension.
        normalize_kernel: Whether to normalize the kernel. If True, the kernel
            is normalized to sum to 1. If False, the kernel is not normalized.
    Note:
        The Gaussian kernel is not an interpolating kernel. It is used for
        smoothing the input data.
    """

    def __init__(
        self,
        truncate_at: Union[Sequence[Tuple[Number, Number]], Tuple[Number, Number], float, int],
        mean: Optional[Union[Sequence[Number], Tensor, int, float]] = None,
        std: Optional[Union[Sequence[Number], Tensor, int, float]] = None,
        normalize_kernel: bool = False,
    ) -> None:
        self._truncate_at = truncate_at
        self._mean = 0.0 if mean is None else mean
        self._std = 0.0 if std is None else std
        self._normalize_kernel = normalize_kernel

    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return False

    def edge_continuity_schedule(self, spatial_dim: int, device: torch_device) -> Tensor:
        return zeros((1, 2), dtype=torch_bool, device=device)

    def piece_edges(self, spatial_dim: int, dtype: torch_dtype, device: torch_device) -> Tensor:
        truncate_at = np.array(self._truncate_at)
        if truncate_at.ndim == 0:
            truncate_at_min = -truncate_at
            truncate_at_max = truncate_at
        elif truncate_at.ndim == 1:
            truncate_at_min, truncate_at_max = truncate_at
        elif truncate_at.ndim == 2:
            truncate_at_min, truncate_at_max = truncate_at[spatial_dim]
        else:
            raise ValueError("Invalid truncation specification")
        return linspace(
            float(truncate_at_min),
            float(truncate_at_max),
            2,
            dtype=dtype,
            device=device,
        )

    def evaluate(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        non_blocking = coordinates.device.type != "cpu"
        if not isinstance(self._mean, Tensor):
            mean = tensor(self._mean, dtype=coordinates.dtype).to(
                device=coordinates.device, non_blocking=non_blocking
            )
        else:
            mean = self._mean
        if not isinstance(self._std, Tensor):
            std = tensor(self._std, dtype=coordinates.dtype).to(
                device=coordinates.device, non_blocking=non_blocking
            )
        else:
            std = self._std
        if mean.ndim == 0:
            mean_dim = mean
        else:
            mean_dim = mean[spatial_dim]
        if std.ndim == 0:
            std_dim = std
        else:
            std_dim = std[spatial_dim]
        values = exp(-((coordinates - mean_dim) ** 2) / (2 * std_dim**2)) / (
            std[spatial_dim] * sqrt(2 * pi)
        )
        if self._normalize_kernel:
            values = values / values.sum()
        return values

    def derivative(self, spatial_dim: int) -> PiecewiseKernelDefinition:
        if self._normalize_kernel:
            raise NotImplementedError(
                "Derivative of normalized Gaussian kernel is not implemented "
                "(normalization brakes the generic implementation)."
            )
        return super().derivative(spatial_dim)


class GaussianSampler(SeparableSampler):
    """Smoothing with Gaussian kernel.

    Arguments:
        truncate_at: Truncation of the kernel. Can be a single value, tuple of
            min and max, or a list of tuples of min and maxf for each spatial
            dimension.
        mean: Mean of the Gaussian kernel. Can be a single value, or a separate
            value for each spatial dimension.
        std: Standard deviation of the Gaussian kernel. Can be a single value,
            or a separate value for each spatial dimension.
        normalize_kernel: Whether to normalize the kernel. If True, the kernel
            is normalized to sum to 1. If False, the kernel is not normalized.
        **kwargs: See `.separable_sampler.SeparableSampler`.
    Note:
        The Gaussian kernel is not an interpolating kernel. It is used for
        smoothing the input data.
    """

    def __init__(
        self,
        truncate_at: Union[Sequence[Tuple[Number, Number]], Tuple[Number, Number], float, int],
        mean: Optional[Union[Sequence[Number], Tensor, int, float]] = None,
        std: Optional[Union[Sequence[Number], Tensor, int, float]] = None,
        normalize_kernel: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            kernel=GaussianKernel(
                truncate_at=truncate_at,
                mean=mean,
                std=std,
                normalize_kernel=normalize_kernel,
            ),
            **kwargs,
        )
