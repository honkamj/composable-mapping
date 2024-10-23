"""Tests for the sampler module."""

from abc import abstractmethod
from typing import Iterable, List
from unittest import TestCase

from torch import Tensor
from torch import device as torch_device
from torch import float32, ones, randn, tensor
from torch.testing import assert_close

from composable_mapping.affine import Affine
from composable_mapping.coordinate_system import CoordinateSystem
from composable_mapping.interface import ISampler
from composable_mapping.mappable_tensor.affine_transformation import (
    HostAffineTransformation,
)
from composable_mapping.mappable_tensor.mappable_tensor import (
    MappableTensor,
    PlainTensor,
    VoxelGrid,
)
from composable_mapping.sampler import (
    BicubicInterpolator,
    LinearInterpolator,
    NearestInterpolator,
)


class ICountingInterpolator(ISampler):
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

    def sample_values(self, values: Tensor, voxel_coordinates: Tensor) -> Tensor:
        self._calls += 1
        return super().sample_values(values, voxel_coordinates)


class CountingNearestInterpolator(ICountingInterpolator, NearestInterpolator):
    """Nearest interpolator that counts the number of calls to the core interpolator"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def sample_values(self, values: Tensor, voxel_coordinates: Tensor) -> Tensor:
        self._calls += 1
        return super().sample_values(values, voxel_coordinates)


class CountingBicubicInterpolator(ICountingInterpolator, BicubicInterpolator):
    """Bicubic interpolator that counts the number of calls to the core interpolator"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def sample_values(self, values: Tensor, voxel_coordinates: Tensor) -> Tensor:
        self._calls += 1
        return super().sample_values(values, voxel_coordinates)


class InterpolatorTest(TestCase):
    """Tests for interpolators"""

    LINEAR_INTERPOLATORS: List[ICountingInterpolator] = [
        CountingLinearInterpolator(extrapolation_mode="border"),
        CountingLinearInterpolator(extrapolation_mode="zeros"),
        CountingLinearInterpolator(extrapolation_mode="reflection"),
    ]

    NEAREST_INTERPOLATORS: List[ICountingInterpolator] = [
        CountingNearestInterpolator(extrapolation_mode="border"),
        CountingNearestInterpolator(extrapolation_mode="zeros"),
        CountingNearestInterpolator(extrapolation_mode="reflection"),
    ]

    BICUBIC_INTERPOLATORS: List[ICountingInterpolator] = [
        CountingBicubicInterpolator(extrapolation_mode="border"),
        CountingBicubicInterpolator(extrapolation_mode="zeros"),
        CountingBicubicInterpolator(extrapolation_mode="reflection"),
    ]

    INTERPOLATORS: List[ICountingInterpolator] = (
        LINEAR_INTERPOLATORS + NEAREST_INTERPOLATORS + BICUBIC_INTERPOLATORS
    )

    def test_voxel_grid_consistency(self):
        """Test interpolator with a voxel grid"""
        grid = VoxelGrid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume,
            grid,
            self.LINEAR_INTERPOLATORS + self.NEAREST_INTERPOLATORS,
            test_mask=True,
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume,
            grid,
            self.BICUBIC_INTERPOLATORS,
            test_mask=False,
        )

    def test_positive_slightly_shifted_voxel_grid_consistency(self):
        """Test interpolator with a slightly shifted voxel grid"""
        grid = (
            CoordinateSystem.voxel(
                spatial_shape=(3, 4),
                voxel_size=(2.0, 2.0),
                dtype=float32,
                device=torch_device("cpu"),
            )
            .shift_voxel(0.01)
            .grid()
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume,
            grid,
            self.LINEAR_INTERPOLATORS + self.NEAREST_INTERPOLATORS,
            test_mask=True,
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume,
            grid,
            self.BICUBIC_INTERPOLATORS,
            test_mask=False,
        )

    def test_negative_slightly_shifted_voxel_grid_consistency(self):
        """Test interpolator with a slightly shifted voxel grid"""
        grid = (
            CoordinateSystem.voxel(
                spatial_shape=(3, 4),
                voxel_size=(2.0, 2.0),
                dtype=float32,
                device=torch_device("cpu"),
            )
            .shift_voxel(-0.01)
            .grid()
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume,
            grid,
            self.INTERPOLATORS,
            test_mask=True,
        )

    def test_strided_grid_consistency(self):
        """Test interpolator with a strided voxel grid"""
        grid = CoordinateSystem.voxel(
            spatial_shape=(3, 4),
            voxel_size=(2.0, 2.0),
            dtype=float32,
            device=torch_device("cpu"),
        ).grid()
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume,
            grid,
            self.LINEAR_INTERPOLATORS + self.NEAREST_INTERPOLATORS,
            test_mask=True,
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume,
            grid,
            self.BICUBIC_INTERPOLATORS,
            test_mask=False,
        )

    def test_strided_and_shifted_grid_consistency(self):
        """Test interpolator with a strided and shited grid"""
        grid = (
            CoordinateSystem.voxel(
                spatial_shape=(3, 4),
                voxel_size=(2.0, 2.0),
                dtype=float32,
                device=torch_device("cpu"),
            )
            .shift_voxel(0.3)
            .grid()
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid, self.INTERPOLATORS)

    def test_strided_and_shifted_grid_consistency_with_extrapolation(self):
        """Test interpolator with an extrapolating grid"""
        grid = (
            CoordinateSystem.voxel(
                spatial_shape=(3, 4),
                voxel_size=(2.0, 2.0),
                dtype=float32,
                device=torch_device("cpu"),
            )
            .shift_voxel(-2.49)
            .grid()
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume, grid, self.INTERPOLATORS, require_conv_interpolation=False
        )

    def test_permuted_grid_consistency(self):
        """Test interpolator with a permuted grid"""
        affine = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [0.0, 1.0, 0.4],
                    [2.0, 0.0, 1.7],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        grid = Affine(affine)(
            VoxelGrid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid, self.INTERPOLATORS)

    def test_flipped_grid_consistency(self):
        """Test interpolator with a permuted flipped grid"""
        affine = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [1.0, 0.0, 6.0],
                    [0.0, -1.0, 8.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        grid = Affine(affine)(
            VoxelGrid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid, self.INTERPOLATORS)

    def test_permuted_flipped_grid_consistency(self):
        """Test interpolator with a permuted flipped grid"""
        affine = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [0.0, 1.0, 6.1],
                    [-3.0, 0.0, 7.2],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        grid = Affine(affine)(
            VoxelGrid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid, self.INTERPOLATORS)

    def test_upsampling_consistency(self):
        """Test interpolator with a voxel grid"""
        grid = (
            CoordinateSystem.voxel(
                spatial_shape=(3, 4),
                voxel_size=(1.0, 1.0),
                dtype=float32,
                device=torch_device("cpu"),
            )
            .reformat(upsampling_factor=(3, 2))
            .shift_voxel(0.8)
            .grid()
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume,
            grid,
            self.INTERPOLATORS,
            test_mask=True,
        )

    def test_upsampling_consistency_with_extrapolation(self):
        """Test interpolator with a voxel grid"""
        grid = (
            CoordinateSystem.voxel(
                spatial_shape=(8, 9),
                voxel_size=(1.0, 1.0),
                dtype=float32,
                device=torch_device("cpu"),
            )
            .reformat(upsampling_factor=(2, 3))
            .shift_voxel(-3)
            .grid()
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(
            test_volume,
            grid,
            self.INTERPOLATORS,
            test_mask=True,
        )

    def test_permuted_flipped_upsampling_consistency(self):
        """Test interpolator with upsampling permuted flipped grid"""
        affine = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [0.0, 1 / 6.0, 6.1],
                    [-2.0, 0.0, 7.2],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        grid = Affine(affine)(
            VoxelGrid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = PlainTensor(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid, self.INTERPOLATORS)

    def _test_grid_interpolation_consistency_with_inputs(
        self,
        volume: MappableTensor,
        grid: MappableTensor,
        interpolators: Iterable[ICountingInterpolator],
        require_conv_interpolation: bool = True,
        test_mask: bool = True,
    ):
        for interpolator in interpolators:
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
                if test_mask:
                    assert_close(
                        conv_interpolated.generate_mask(generate_missing_mask=True),
                        grid_sample_interpolated.generate_mask(generate_missing_mask=True),
                        check_layout=False,
                    )
