"""Tests for the interpolator module."""

from abc import abstractmethod
from typing import List
from unittest import TestCase

from torch import Tensor
from torch import device as torch_device
from torch import float32, ones, randn, tensor
from torch.testing import assert_close

from composable_mapping.affine import Affine
from composable_mapping.coordinate_system import CoordinateSystem
from composable_mapping.interface import IInterpolator
from composable_mapping.interpolator import LinearInterpolator
from composable_mapping.mappable_tensor.affine_transformation import (
    HostAffineTransformation,
)
from composable_mapping.mappable_tensor.mappable_tensor import (
    MappableTensor,
    PlainTensor,
    VoxelGrid,
)


class ICountingInterpolator(IInterpolator):
    """Linear interpolator that counts the number of calls to the core interpolator"""

    @property
    @abstractmethod
    def calls(self) -> int:
        """Number of calls to the interpolator"""


class CountingLinearInterpolator(ICountingInterpolator, LinearInterpolator):
    """Linear interpolator that counts the number of calls to the core interpolator"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def interpolate_values(self, values: Tensor, voxel_coordinates: Tensor) -> Tensor:
        self._calls += 1
        return super().interpolate_values(values, voxel_coordinates)


class InterpolatorTest(TestCase):
    """Tests for linear interpolator"""

    INTERPOLATORS: List[ICountingInterpolator] = [
        # CountingLinearInterpolator(extrapolation_mode="border"),
        CountingLinearInterpolator(extrapolation_mode="zeros"),
        # CountingLinearInterpolator(extrapolation_mode="reflection"),
        # NearestInterpolator(extrapolation_mode="border"),
        # BicubicInterpolator(extrapolation_mode="border"),
    ]

    def test_voxel_grid_consistency(self):
        """Test interpolator with a voxel grid"""
        grid = VoxelGrid(spatial_shape=(3, 4, 5), dtype=float32, device=torch_device("cpu"))
        mask = ones((2, 1, 14, 15, 16), dtype=float32, device=torch_device("cpu"))
        mask[:, :, :4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15, 16), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid)

    def test_strided_grid_consistency(self):
        """Test interpolator with a strided voxel grid"""
        grid = CoordinateSystem.voxel(
            spatial_shape=(3, 4, 5),
            voxel_size=(2.0, 2.0, 2.0),
            dtype=float32,
            device=torch_device("cpu"),
        ).grid()
        mask = ones((2, 1, 14, 15, 16), dtype=float32, device=torch_device("cpu"))
        mask[:, :, :4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15, 16), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid)

    def test_strided_and_shifted_grid_consistency(self):
        """Test interpolator with a strided and shited grid"""
        grid = (
            CoordinateSystem.voxel(
                spatial_shape=(3, 4, 5),
                voxel_size=(2.0, 2.0, 2.0),
                dtype=float32,
                device=torch_device("cpu"),
            )
            .shift_voxel(0.3)
            .grid()
        )
        mask = ones((2, 1, 14, 15, 16), dtype=float32, device=torch_device("cpu"))
        mask[:, :, :4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15, 16), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid)

    def test_strided_and_shifted_grid_consistency_with_extrapolation(self):
        """Test interpolator with an extrapolating grid"""
        grid = (
            CoordinateSystem.voxel(
                spatial_shape=(3, 4, 5),
                voxel_size=(2.0, 2.0, 2.0),
                dtype=float32,
                device=torch_device("cpu"),
            )
            .shift_voxel(-2.5)
            .grid()
        )
        mask = ones((2, 1, 14, 15, 16), dtype=float32, device=torch_device("cpu"))
        mask[:, :, :4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15, 16), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume, grid, require_conv_interpolation=False
        )

    def test_permuted_grid_consistency(self):
        """Test interpolator with a permuted grid"""
        affine = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [0.0, 1.0, 0.0, 0.4],
                    [2.0, 0.0, 0.0, 1.7],
                    [0.0, 0.0, 1.0, 2.5],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        grid = Affine(affine)(
            VoxelGrid(spatial_shape=(3, 4, 5), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15, 16), dtype=float32, device=torch_device("cpu"))
        mask[:, :, :4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15, 16), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid)

    def test_flipped_grid_consistency(self):
        """Test interpolator with a permuted flipped grid"""
        affine = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [1.0, 0.0, 0.0, 6.0],
                    [0.0, 1.0, 0.0, 7.0],
                    [0.0, 0.0, -1.0, 8.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        grid = Affine(affine)(
            VoxelGrid(spatial_shape=(3, 4, 5), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15, 16), dtype=float32, device=torch_device("cpu"))
        mask[:, :, :4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15, 16), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid)

    def test_permuted_flipped_grid_consistency(self):
        """Test interpolator with a permuted flipped grid"""
        affine = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [0.0, 1.0, 0.0, 6.1],
                    [-3.0, 0.0, 0.0, 7.2],
                    [0.0, 0.0, -1.0, 8.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        grid = Affine(affine)(
            VoxelGrid(spatial_shape=(3, 4, 5), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15, 16), dtype=float32, device=torch_device("cpu"))
        mask[:, :, :4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15, 16), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid)

    def _test_grid_interpolation_consistency_with_inputs(
        self, volume: MappableTensor, grid: MappableTensor, require_conv_interpolation: bool = True
    ):
        for interpolator in self.INTERPOLATORS:
            with self.subTest(interpolator=interpolator):
                grid_sample_calls = interpolator.calls
                conv_interpolated = interpolator(volume, coordinates=grid)
                if require_conv_interpolation:
                    self.assertEqual(interpolator.calls, grid_sample_calls)
                grid_sample_calls = interpolator.calls
                grid_sample_interpolated = interpolator(volume, coordinates=grid.reduce())
                self.assertEqual(interpolator.calls, grid_sample_calls + 1)
                assert_close(
                    conv_interpolated.generate_values(),
                    grid_sample_interpolated.generate_values(),
                    check_layout=False,
                )
                assert_close(
                    conv_interpolated.generate_mask(generate_missing_mask=True),
                    grid_sample_interpolated.generate_mask(generate_missing_mask=True),
                    check_layout=False,
                )
