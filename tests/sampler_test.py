"""Tests for the sampler module."""

from abc import abstractmethod
from typing import Iterable, List
from unittest import TestCase

from torch import Tensor
from torch import device as torch_device
from torch import eye, float32, manual_seed, ones, randint, randn, randperm, tensor
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
from composable_mapping.affine_transformation import (
    AffineTransformation,
    HostAffineTransformation,
)
from composable_mapping.affine_transformation.matrix import embed_matrix
from composable_mapping.sampler.base import (
    BaseSeparableSampler,
    ISeparableKernelSupport,
    SymmetricPolynomialKernelSupport,
)
from composable_mapping.sampler.convolution_sampling import (
    apply_flipping_permutation_to_affine_matrix,
    apply_flipping_permutation_to_volume,
    obtain_normalizing_flipping_permutation,
)
from composable_mapping.sampler.interface import LimitDirection


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


class _NonSymmetricInterpolator(BaseSeparableSampler):

    def __init__(
        self,
    ) -> None:
        super().__init__(
            extrapolation_mode="border",
            mask_extrapolated_regions_for_empty_volume_mask=True,
            convolution_threshold=1e-4,
            mask_threshold=1e-5,
            limit_direction=LimitDirection.left(),
        )

    def _kernel_support(self, spatial_dim: int) -> ISeparableKernelSupport:
        return SymmetricPolynomialKernelSupport(
            kernel_width=[4, 5, 6][spatial_dim],
            polynomial_degree=1,
        )

    def _is_interpolating_kernel(self, spatial_dim: int) -> bool:
        return False

    def _left_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        return coordinates + 1

    def _right_limit_kernel(self, coordinates: Tensor, spatial_dim: int) -> Tensor:
        raise NotImplementedError

    def sample_mask(self, mask: Tensor, coordinates: Tensor) -> Tensor:
        raise NotImplementedError

    def sample_values(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        raise NotImplementedError


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
            .grid()
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
            .grid()
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
        ).grid()
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
            .grid()
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
            .grid()
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
            .grid()
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
            .grid()
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
        interpolated_1 = _NonSymmetricInterpolator()(test_volume, coordinates=grid_1)
        interpolated_2 = _NonSymmetricInterpolator()(test_volume, coordinates=grid_2)
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
        interpolated_1 = _NonSymmetricInterpolator()(test_volume, coordinates=grid_1)
        interpolated_2 = _NonSymmetricInterpolator()(test_volume, coordinates=grid_2)
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


class FlippingPermutationTest(TestCase):
    """Tests for flipping permutation"""

    def test_flipping_permutation(self):
        """Test that one ends in same coordinate with permuting the volume and
        applying the transformation"""
        for _ in range(100):
            spatial_permutation = randperm(5).tolist()
            flipped_spatial_dims = randperm(5)[: randint(high=5, size=tuple()).item()].tolist()
            matrix = eye(5, 6, dtype=float32)
            flipping_permutation_matrix = apply_flipping_permutation_to_affine_matrix(
                matrix=matrix[None],
                spatial_permutation=spatial_permutation,
                flipped_spatial_dims=flipped_spatial_dims,
                volume_spatial_shape=[5, 6, 7, 8, 9],
            )[0]
            transformation = AffineTransformation(embed_matrix(flipping_permutation_matrix, (6, 6)))

            test_value = tensor(
                [
                    randint(high=5, size=tuple(), dtype=float32).item(),
                    randint(high=6, size=tuple(), dtype=float32).item(),
                    randint(high=7, size=tuple(), dtype=float32).item(),
                    randint(high=8, size=tuple(), dtype=float32).item(),
                    randint(high=9, size=tuple(), dtype=float32).item(),
                ],
                dtype=float32,
            )

            grid = voxel_grid(
                (5, 6, 7, 8, 9), dtype=float32, device=torch_device("cpu")
            ).generate_values()

            assert_close(
                grid[(...,) + tuple(test_value.long().tolist())],
                apply_flipping_permutation_to_volume(
                    grid,
                    n_channel_dims=1,
                    spatial_permutation=spatial_permutation,
                    flipped_spatial_dims=flipped_spatial_dims,
                )[(...,) + tuple(transformation(test_value).long().tolist())],
            )

    def test_flipping_permutation_normalization(self):
        """Test that one ends in same coordinate with permuting the volume and
        applying the transformation"""
        for _ in range(100):
            random_permutation = randperm(5)
            matrix = eye(6, dtype=float32)
            random_flips = 2 * randint(0, 2, (5,), dtype=float32) - 1
            matrix[:-1, :-1] = random_flips * matrix[:-1, :-1]
            matrix[:-1, :-1] = matrix[:-1, :-1][random_permutation.long()]
            spatial_permutation, flipped_spatial_dims = obtain_normalizing_flipping_permutation(
                matrix[:-1]
            )
            normalized_matrix = apply_flipping_permutation_to_affine_matrix(
                matrix=matrix[None, :-1],
                spatial_permutation=spatial_permutation,
                flipped_spatial_dims=flipped_spatial_dims,
                volume_spatial_shape=[5, 6, 7, 8, 9],
            )[0]
            normalized_matrix = embed_matrix(normalized_matrix, (6, 6))
            assert_close(
                normalized_matrix[:-1, :-1],
                eye(5, dtype=float32),
            )
