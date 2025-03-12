"""Tests for the sampler module."""

from abc import abstractmethod
from typing import Iterable, List, Sequence, Tuple
from unittest import TestCase

from torch import Tensor
from torch import bool as torch_bool
from torch import cat
from torch import device as torch_device
from torch import dtype as torch_dtype
from torch import float32, linspace, manual_seed, ones, randn, stack, tensor, zeros
from torch.testing import assert_close

from composable_mapping import (
    Affine,
    BicubicInterpolator,
    CoordinateSystem,
    ISampler,
    LinearInterpolator,
    MappableTensor,
    NearestInterpolator,
    mappable,
    voxel_grid,
)
from composable_mapping.affine_transformation import HostAffineTransformation
from composable_mapping.sampler.convolution_sampling import (
    apply_flips_and_permutation_to_volume,
    normalize_sampling_grid,
)
from composable_mapping.sampler.interface import LimitDirection
from composable_mapping.sampler.separable_sampler import (
    PiecewiseKernelDefinition,
    SeparableSampler,
)


class ICountingInterpolator(ISampler):
    """Linear interpolator that counts the number of calls to the core interpolator"""

    @property
    @abstractmethod
    def calls(self) -> int:
        """Number of calls to the interpolator"""


class CountingLinearInterpolator(LinearInterpolator, ICountingInterpolator):
    """Linear interpolator that counts the number of calls to the core interpolator"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def sample_values(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        self._calls += 1
        return super().sample_values(volume, coordinates)


class CountingNearestInterpolator(NearestInterpolator, ICountingInterpolator):
    """Nearest interpolator that counts the number of calls to the core interpolator"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def sample_values(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        self._calls += 1
        return super().sample_values(volume, coordinates)


class CountingBicubicInterpolator(BicubicInterpolator, ICountingInterpolator):
    """Bicubic interpolator that counts the number of calls to the core interpolator"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._calls = 0

    @property
    def calls(self) -> int:
        return self._calls

    def sample_values(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        self._calls += 1
        return super().sample_values(volume, coordinates)


class _NonSymmetricKernel(PiecewiseKernelDefinition):
    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return False

    def edge_continuity_schedule(self, spatial_dim: int, device: torch_device) -> Tensor:
        return stack(
            [
                zeros(2, device=device, dtype=torch_bool),
                ones(2, device=device, dtype=torch_bool),
            ],
            dim=0,
        )

    def piece_edges(self, spatial_dim: int, dtype: torch_dtype, device: torch_device) -> Tensor:
        if spatial_dim == 0:
            return linspace(-2.0, 2.0, steps=2, dtype=dtype, device=device)
        if spatial_dim == 1:
            return linspace(-2.5, 2.5, steps=2, dtype=dtype, device=device)
        if spatial_dim == 2:
            return linspace(-3.0, 3.0, steps=2, dtype=dtype, device=device)
        raise ValueError("Invalid spatial_dim.")

    def evaluate(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        return coordinates + 1


class _ShiftedLinearKernel(PiecewiseKernelDefinition):
    def is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return False

    def edge_continuity_schedule(self, spatial_dim: int, device: torch_device) -> Tensor:
        return stack(
            [
                ones(3, device=device, dtype=torch_bool),  # Original
                zeros(3, device=device, dtype=torch_bool),  # First derivative
                ones(3, device=device, dtype=torch_bool),  # Second derivative and beyond
            ],
            dim=0,
        )

    def piece_edges(self, spatial_dim: int, dtype: torch_dtype, device: torch_device) -> Tensor:
        return linspace(0.0, 2.0, steps=3, dtype=dtype, device=device)

    def evaluate(self, spatial_dim: int, coordinates: Tensor) -> Tensor:
        return stack(
            [
                1 + (coordinates[0, :] - 1),
                1 - (coordinates[1, :] - 1),
            ],
            dim=0,
        )


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
        grid = voxel_grid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
            .translate_voxel(0.01)
            .grid
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
            .translate_voxel(-0.01)
            .grid
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
        ).grid
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
            .translate_voxel(0.3)
            .grid
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
            .translate_voxel(-2.49)
            .grid
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
            voxel_grid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
            voxel_grid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
            voxel_grid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
            .translate_voxel(0.8)
            .grid
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
            .translate_world(-3)
            .grid
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
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
            voxel_grid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
        )
        mask = ones((2, 1, 14, 15), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:4] = 0.0
        test_volume = mappable(
            randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        self._test_grid_interpolation_consistency_with_inputs(test_volume, grid, self.INTERPOLATORS)

    def test_non_central_kernel_consistency(self):
        """Test consistency of non-central kernels with shifted linear kernel"""
        grids = [
            voxel_grid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu")) + 5.2,
            0.25 * voxel_grid(spatial_shape=(3, 4), dtype=float32, device=torch_device("cpu"))
            + 5.2,
        ]
        for grid in grids:
            test_volume = mappable(
                randn((2, 3, 14, 15), dtype=float32, device=torch_device("cpu")),
            )
            non_central_sampler = SeparableSampler(
                kernel=_ShiftedLinearKernel(),
                conv_tol=1e-4,
                mask_tol=1e-5,
            )
            central_sampler = LinearInterpolator(
                extrapolation_mode="border",
                conv_tol=1e-4,
                mask_tol=1e-5,
            )
            interpolated_non_central = non_central_sampler(test_volume, coordinates=grid)
            interpolated_central = central_sampler(test_volume, coordinates=grid + 1)
            assert_close(
                interpolated_non_central.generate_values(),
                interpolated_central.generate_values(),
                check_layout=False,
            )

    def test_permuted_consistency_with_non_symmetric_kernel_with_integer_translation(self):
        """Test that interpolation is consistent with a non-symmetric kernel and
        integer translation"""
        manual_seed(0)
        affine_1 = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [0.0, 0.0, -1.0, 5.0],
                    [1.0, 0.0, 0.0, 5.0],
                    [0.0, -1.0, 0.0, 5.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        affine_2 = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [1.0, 0.0, 0.0, 5.0],
                    [0.0, 1.0, 0.0, 5.0],
                    [0.0, 0.0, 1.0, 5.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        grid_1 = Affine(affine_1)(
            voxel_grid(spatial_shape=(1, 1, 1), dtype=float32, device=torch_device("cpu"))
        )
        grid_2 = Affine(affine_2)(
            voxel_grid(spatial_shape=(1, 1, 1), dtype=float32, device=torch_device("cpu"))
        )
        assert_close(grid_1.generate_values(), grid_2.generate_values(), check_layout=False)
        mask = ones((1, 1, 15, 14, 13), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:5] = 0.0
        test_volume = mappable(
            randn((1, 1, 15, 14, 13), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        non_symmetric_sampler = SeparableSampler(
            kernel=_NonSymmetricKernel(),
            extrapolation_mode="border",
            mask_extrapolated_regions=True,
            conv_tol=1e-4,
            mask_tol=1e-5,
            limit_direction=LimitDirection.left(),
        )
        interpolated_1 = non_symmetric_sampler(test_volume, coordinates=grid_1)
        interpolated_2 = non_symmetric_sampler(test_volume, coordinates=grid_2)
        assert_close(
            interpolated_1.generate_values(), interpolated_2.generate_values(), check_layout=False
        )
        assert_close(
            interpolated_1.generate_mask(), interpolated_2.generate_mask(), check_layout=False
        )

    def test_permuted_consistency_with_non_symmetric_kernel_with_float_translation(self):
        """Test that interpolation is consistent with a non-symmetric kernel and
        float translation"""
        affine_1 = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [0.0, 0.0, -1.0, 5.4],
                    [1.0, 0.0, 0.0, 5.4],
                    [0.0, -1.0, 0.0, 5.4],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        affine_2 = HostAffineTransformation(
            transformation_matrix_on_host=tensor(
                [
                    [1.0, 0.0, 0.0, 5.4],
                    [0.0, 1.0, 0.0, 5.4],
                    [0.0, 0.0, 1.0, 5.4],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float32,
            ),
        )
        grid_1 = Affine(affine_1)(
            voxel_grid(spatial_shape=(1, 1, 1), dtype=float32, device=torch_device("cpu"))
        )
        grid_2 = Affine(affine_2)(
            voxel_grid(spatial_shape=(1, 1, 1), dtype=float32, device=torch_device("cpu"))
        )
        assert_close(grid_1.generate_values(), grid_2.generate_values(), check_layout=False)
        mask = ones((1, 1, 15, 14, 13), dtype=float32, device=torch_device("cpu"))
        mask[:, :, 2:5] = 0.0
        test_volume = mappable(
            randn((1, 1, 15, 14, 13), dtype=float32, device=torch_device("cpu")), mask=mask
        )
        non_symmetric_sampler = SeparableSampler(
            kernel=_NonSymmetricKernel(),
            extrapolation_mode="border",
            mask_extrapolated_regions=True,
            conv_tol=1e-4,
            mask_tol=1e-5,
            limit_direction=LimitDirection.left(),
        )
        interpolated_1 = non_symmetric_sampler(test_volume, coordinates=grid_1)
        interpolated_2 = non_symmetric_sampler(test_volume, coordinates=grid_2)
        assert_close(
            interpolated_1.generate_values(),
            interpolated_2.generate_values(),
            check_layout=False,
            atol=1e-4,
            rtol=1e-5,
        )
        assert_close(
            interpolated_1.generate_mask(), interpolated_2.generate_mask(), check_layout=False
        )

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


class NormalizeSamplingGridTest(TestCase):
    """Tests for normalizing sampling grid"""

    def test_normalize_sampling_grid(self):
        """Test normalize_sampling_grid"""
        affine_matrix = tensor(
            [
                [0, -2.0, 0.0, -5.2],
                [3.0, 0.0, 0.0, 3.7],
                [0.0, 0.0, -1.0, 2.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        grid_shape = (3, 4, 5)
        (
            normalized_grid_shape,
            normalized_affine_matrix,
            inverse_spatial_permutation,
            flipped_spatial_dims,
        ) = normalize_sampling_grid(grid_shape, affine_matrix)

        assert_close(
            normalized_affine_matrix[0, :, :-1].diag().diag(), normalized_affine_matrix[0, :, :-1]
        )
        assert_close(normalized_affine_matrix[0, :, :-1].abs(), normalized_affine_matrix[0, :, :-1])

        grid_1 = Affine.from_matrix(affine_matrix)(voxel_grid(grid_shape)).generate_values()
        grid_2 = Affine.from_matrix(
            cat((normalized_affine_matrix, tensor([0.0, 0.0, 0.0, 1.0])[None, None]), dim=1)
        )(voxel_grid(normalized_grid_shape)).generate_values()

        back_transformed_grid_2 = apply_flips_and_permutation_to_volume(
            grid_2,
            n_channel_dims=1,
            spatial_permutation=inverse_spatial_permutation,
            flipped_spatial_dims=flipped_spatial_dims,
        )
        assert_close(grid_1, back_transformed_grid_2)
