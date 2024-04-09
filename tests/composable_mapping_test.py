"""Tests for composable mappings"""

from unittest import TestCase

from deformation_inversion_layer.interpolator import LinearInterpolator
from torch import Tensor, matmul, tensor
from torch.testing import assert_close

from composable_mapping.affine import (
    AffineTransformation,
    CPUAffineTransformation,
    convert_to_homogenous_coordinates,
)
from composable_mapping.coordinate_system_factory import (
    create_centered_normalized,
    create_top_left_aligned,
    create_top_left_aligned_normalized,
)
from composable_mapping.dense_deformation import generate_voxel_coordinate_grid
from composable_mapping.grid_mapping import (
    GridCoordinateMapping,
    GridMappingArgs,
    GridVolume,
)
from composable_mapping.mapping_factory import GridComposableFactory
from composable_mapping.masked_tensor import MaskedTensor, VoxelCoordinateGrid


class _CountingInterpolator(LinearInterpolator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.counter = 0

    def __call__(self, volume: Tensor, coordinates: Tensor) -> Tensor:
        self.counter += 1
        return super().__call__(volume, coordinates)


class ComposableMappingTests(TestCase):
    """Tests for composable mappings"""

    def test_affine_composition(self) -> None:
        """Test that affine composition works correctly"""
        matrix_1 = tensor(
            [
                [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        matrix_2 = tensor(
            [
                [[2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
                [[2.0, 0.0, 0.0], [0.0, 5.0, -1.0], [0.0, 0.0, 1.0]],
            ]
        )
        input_vector = tensor([[-5.0, -2.0], [3.0, 2.0]])
        expected_output = matmul(
            matmul(matrix_2, matrix_1), convert_to_homogenous_coordinates(input_vector)[..., None]
        )[..., :-1, 0]
        transformation_1 = AffineTransformation(matrix_1)
        transformation_2 = AffineTransformation(matrix_2)
        composition = transformation_2.compose(transformation_1)
        assert_close(composition(input_vector), expected_output)
        assert_close(transformation_2(transformation_1(input_vector)), expected_output)

    def test_cpu_affine_composition(self) -> None:
        """Test that affine composition works correctly"""
        matrix_1 = tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        matrix_2 = tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        input_vector = tensor([[-5.0, -2.0], [3.0, 2.0]])
        expected_output = matmul(
            matmul(matrix_2, matrix_1), convert_to_homogenous_coordinates(input_vector)[..., None]
        )[..., :-1, 0]
        expected_inverse_output = matmul(
            matrix_1.inverse(), convert_to_homogenous_coordinates(input_vector)[..., None]
        )[..., :-1, 0]
        cpu_composable_1 = CPUAffineTransformation(matrix_1)
        cpu_composable_2 = CPUAffineTransformation(matrix_2)
        lazy_inverse_1 = cpu_composable_1.invert()
        assert_close(cpu_composable_2.compose(cpu_composable_1)(input_vector), expected_output)
        assert_close(cpu_composable_2(cpu_composable_1(input_vector)), expected_output)
        assert_close(lazy_inverse_1(input_vector), expected_inverse_output)

    def test_non_cpu_and_cpu_affine_composition(self) -> None:
        """Test that affine composition works correctly"""
        matrix_1 = tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        matrix_2 = tensor(
            [
                [[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        input_vector = tensor([[-5.0, -2.0], [3.0, 2.0]])
        expected_output = matmul(
            matmul(matrix_2, matrix_1), convert_to_homogenous_coordinates(input_vector)[..., None]
        )[..., :-1, 0]
        cpu_composable_1 = CPUAffineTransformation(matrix_1)
        transformation_2 = AffineTransformation(matrix_2)
        composition = transformation_2.compose(cpu_composable_1)
        assert_close(composition(input_vector), expected_output)
        assert_close(transformation_2(cpu_composable_1(input_vector)), expected_output)

    def test_grid_volume(self) -> None:
        """Test that grid volumes work correctly"""
        data = tensor(
            [
                [[1.0, 0.0, 5.0], [0.0, -2.0, 0.0], [2.0, -3.0, 4.0], [2.0, 0.0, 1.0]],
                [[1.0, 1.0, 1.0], [0.0, 4.0, 0.0], [-1.0, 5.0, 3.0], [-2.0, 0.0, 1.0]],
            ]
        )[None]
        mask = tensor([[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])[None]
        interpolator = _CountingInterpolator(padding_mode="border")
        volume = GridVolume(
            data=data,
            mask=mask,
            n_channel_dims=1,
            grid_mapping_args=GridMappingArgs(interpolator=interpolator, mask_outside_fov=True),
        )
        input_points = (
            tensor([1.0, 1.0]),
            tensor([2.5, 2.0])[None, ..., None, None],
            tensor([2.5, 2.01])[None, ..., None, None],
        )
        output_points = (
            tensor([-2.0, 4.0])[None],
            tensor([2.5, 2.0])[None, ..., None, None],
            tensor([2.5, 2.0])[None, ..., None, None],
        )
        output_masks = (
            tensor([0.0])[None],
            tensor([1.0])[None, ..., None, None],
            tensor([0.0])[None, ..., None, None],
        )
        for input_point, expected_output, expected_mask in zip(
            input_points, output_points, output_masks
        ):
            output = volume(MaskedTensor(input_point))
            assert_close(output.generate_values(), expected_output)
            assert_close(output.generate_mask(), expected_mask)
        count_before = interpolator.counter
        assert_close(data, volume(VoxelCoordinateGrid((4, 3))).generate_values())
        self.assertEqual(interpolator.counter, count_before)

    def test_grid_mapping(self) -> None:
        """Test that grid volumes work correctly"""
        data = tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0], [-2.0, -2.0, -2.0]],
            ]
        )[None]
        mask = tensor([[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])[None]
        interpolator = _CountingInterpolator(padding_mode="border")
        mapping = GridCoordinateMapping(
            displacement_field=data,
            mask=mask,
            grid_mapping_args=GridMappingArgs(
                interpolator=interpolator,
                mask_outside_fov=True,
            ),
        )
        input_points = (
            tensor([0.3, 1.5]),
            tensor([2.0, 1.5])[None, ..., None, None],
            tensor([2.0, 2.01])[None, ..., None, None],
        )
        output_points = (
            tensor([0.3, 1.5])[None],
            tensor([3.0, -0.5])[None, ..., None, None],
            tensor([3.0, 0.01])[None, ..., None, None],
        )
        output_masks = (
            tensor([0.0])[None],
            tensor([1.0])[None, ..., None, None],
            tensor([0.0])[None, ..., None, None],
        )
        for input_point, expected_output, expected_mask in zip(
            input_points, output_points, output_masks
        ):
            output = mapping(MaskedTensor(input_point))
            assert_close(output.generate_values(), expected_output)
            assert_close(output.generate_mask(), expected_mask)
        count_before = interpolator.counter
        assert_close(
            data + generate_voxel_coordinate_grid((4, 3)),
            mapping(VoxelCoordinateGrid((4, 3))).generate_values(),
        )
        self.assertEqual(interpolator.counter, count_before)

    def test_slice_generation_for_voxel_grids(self) -> None:
        """Test that correct slices are generated"""
        voxel_grid_1 = VoxelCoordinateGrid(
            (10, 20),
            affine_transformation=CPUAffineTransformation(
                tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
            ),
        )
        self.assertEqual(voxel_grid_1.as_slice((30, 30)), (..., slice(2, 12, 1), slice(3, 23, 1)))
        voxel_grid_2 = VoxelCoordinateGrid(
            (10, 20),
            affine_transformation=CPUAffineTransformation(
                tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            ),
        )
        self.assertEqual(voxel_grid_2.as_slice((10, 20)), (..., slice(0, 10, 1), slice(0, 20, 1)))


class ComposableFactoryTests(TestCase):
    """Test different mappings together"""

    def test_coordinate_transformed_grid_volume(self) -> None:
        """Test coordinate transformed grid volume"""
        data = tensor(
            [
                [[1.0, 0.0, 5.0], [0.0, -2.0, 0.0], [2.0, -3.0, 4.0], [2.0, 0.0, 1.0]],
                [[1.0, 1.0, 1.0], [0.0, 4.0, 0.0], [-1.0, 5.0, 3.0], [-2.0, 0.0, 1.0]],
            ]
        )[None]
        mask = tensor([[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])[None]
        coordinate_system = create_centered_normalized(
            original_grid_shape=(4, 3), original_voxel_size=(1.0, 2.0)
        )
        interpolator = _CountingInterpolator(padding_mode="border")
        volume = GridComposableFactory(
            coordinate_system=coordinate_system,
            grid_mapping_args=GridMappingArgs(
                interpolator=interpolator, mask_outside_fov=True, mask_threshold=1.0
            ),
        ).create_volume(
            data=data,
            mask=mask,
            n_channel_dims=1,
        )
        input_points = (tensor([-1 / 2, 0.0]), tensor([-1 / 6, -1 / 3])[None, ..., None, None])
        output_points = (tensor([0.0, 1.0])[None], tensor([-1.0, 2.0])[None, ..., None, None])
        output_masks = (tensor([1.0])[None], tensor([0.0])[None, ..., None, None])
        for input_point, expected_output, expected_mask in zip(
            input_points, output_points, output_masks
        ):
            output = volume(MaskedTensor(input_point))
            assert_close(output.generate_values(), expected_output)
            assert_close(output.generate_mask(), expected_mask)
        middle_coordinate_system = create_centered_normalized(
            original_grid_shape=(4, 3), original_voxel_size=(1.0, 2.0), grid_shape=(2, 3)
        )
        count_before = interpolator.counter
        assert_close(
            data[:, :, 1:-1],
            volume(middle_coordinate_system.grid).generate_values(),
        )
        self.assertEqual(interpolator.counter, count_before)

    def test_coordinate_transformed_grid_mapping(self) -> None:
        """Test that grid mappings work correctly"""
        data = tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0], [-2.0, -2.0, -2.0]],
            ]
        )[None]
        mask = tensor([[[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]])[None]
        interpolator = _CountingInterpolator(padding_mode="border")
        coordinate_system = create_centered_normalized(
            original_grid_shape=(4, 3), original_voxel_size=(1.0, 2.0)
        )
        mapping = GridComposableFactory(
            coordinate_system=coordinate_system,
            grid_mapping_args=GridMappingArgs(
                interpolator=interpolator, mask_outside_fov=True, mask_threshold=1.0
            ),
        ).create_deformation(
            displacement_field=data,
            mask=mask,
        )
        input_points = (
            tensor([-1 / 2, 0.0]),
            tensor([-1 / 2 + 1e-4, 0.0]),
            tensor([1 / 6, 1 / 3])[None, ..., None, None],
        )
        output_points = (
            tensor([-1 / 2, 0.0])[None],
            tensor([-1 / 2 + 1e-4, 0.0])[None],
            tensor([1 / 2, -1])[None, ..., None, None],
        )
        output_masks = (
            tensor([1.0])[None],
            tensor([0.0])[None],
            tensor([1.0])[None, ..., None, None],
        )
        for input_point, expected_output, expected_mask in zip(
            input_points, output_points, output_masks
        ):
            output = mapping(MaskedTensor(input_point))
            assert_close(output.generate_values(), expected_output)
            assert_close(output.generate_mask(), expected_mask)
        middle_coordinate_system = create_centered_normalized(
            original_grid_shape=(4, 3), original_voxel_size=(1.0, 2.0), grid_shape=(2, 3)
        )
        count_before = interpolator.counter
        grid_values = mapping(middle_coordinate_system.grid)
        assert_close(
            (data + generate_voxel_coordinate_grid((4, 3), data.device))[..., 1:-1, :],
            coordinate_system.to_voxel_coordinates(grid_values).generate_values(),
        )
        self.assertEqual(interpolator.counter, count_before)

    def test_centered_normalized_coordinate_system_consistency(self) -> None:
        """Test that coordinate system is consistent"""
        shape = (5, 6)
        coordinate_systems = [
            create_centered_normalized(
                original_grid_shape=(13, 4), original_voxel_size=(3.0, 2.05), grid_shape=shape
            ),
            create_centered_normalized(
                original_grid_shape=(13, 4),
                original_voxel_size=(3.0, 2.05),
                grid_shape=shape,
                downsampling_factor=(1.2, 0.54, 17.3),
            ),
            create_centered_normalized(original_grid_shape=shape, original_voxel_size=(1.0, 2.05)),
        ]
        for coordinate_system in coordinate_systems:
            voxel_grid = generate_voxel_coordinate_grid(shape)
            assert_close(
                coordinate_system.to_voxel_coordinates(coordinate_system.grid).generate_values(),
                voxel_grid,
            )
            assert_close(
                coordinate_system.from_voxel_coordinates(
                    coordinate_system.to_voxel_coordinates(coordinate_system.grid)
                ).generate_values(),
                coordinate_system.grid.generate_values(),
            )

    def test_top_left_aligned_normalized_coordinate_system_consistency(self) -> None:
        """Test that coordinate system is consistent"""
        shape = (5, 6)
        original_grid_shape = (16, 18)
        coordinate_systems = [
            create_top_left_aligned_normalized(
                original_grid_shape=original_grid_shape,
                grid_shape=shape,
                original_voxel_size=(3.0, 2.05),
                downsampling_factor=[1.0, 0.7],
            ),
            create_top_left_aligned_normalized(
                original_grid_shape=original_grid_shape,
                grid_shape=shape,
                original_voxel_size=(1.0, 2.05),
                downsampling_factor=[2.0, 4.0],
            ),
        ]
        for coordinate_system in coordinate_systems:
            voxel_grid = generate_voxel_coordinate_grid(shape)
            assert_close(
                coordinate_system.to_voxel_coordinates(coordinate_system.grid).generate_values(),
                voxel_grid,
            )
            assert_close(
                coordinate_system.from_voxel_coordinates(
                    coordinate_system.to_voxel_coordinates(coordinate_system.grid)
                ).generate_values(),
                coordinate_system.grid.generate_values(),
            )

    def test_top_left_aligned_coordinate_system_consistency(self) -> None:
        """Test that coordinate system is consistent"""
        shape = (5, 6)
        original_grid_shape = (16, 18)
        coordinate_systems = [
            create_top_left_aligned(
                original_grid_shape=original_grid_shape,
                grid_shape=shape,
                original_voxel_size=(3.0, 2.05),
                downsampling_factor=[1.0, 0.7],
            ),
            create_top_left_aligned(
                original_grid_shape=original_grid_shape,
                grid_shape=shape,
                original_voxel_size=(1.0, 2.05),
                downsampling_factor=[2.0, 4.0],
            ),
        ]
        for coordinate_system in coordinate_systems:
            voxel_grid = generate_voxel_coordinate_grid(shape)
            assert_close(
                coordinate_system.to_voxel_coordinates(coordinate_system.grid).generate_values(),
                voxel_grid,
            )
            assert_close(
                coordinate_system.from_voxel_coordinates(
                    coordinate_system.to_voxel_coordinates(coordinate_system.grid)
                ).generate_values(),
                coordinate_system.grid.generate_values(),
            )

    def test_top_left_aligned_normalized_and_centered_normalized_coordinate_system_consistency(
        self,
    ) -> None:
        """Test that coordinate systems are consistent between themselves"""
        centered_coordinate_systems = [
            create_centered_normalized(
                original_grid_shape=(17, 14), original_voxel_size=(1.2, 2.05)
            ),
            create_centered_normalized(original_grid_shape=(8, 6), original_voxel_size=(3.1, 2.7)),
            create_centered_normalized(original_grid_shape=(2, 3), original_voxel_size=(3.0, 1.0)),
        ]
        top_left_aligned_coordinate_systems = [
            create_top_left_aligned_normalized(
                original_grid_shape=(17, 14),
                grid_shape=(17, 14),
                original_voxel_size=(1.2, 2.05),
                downsampling_factor=[1.0, 1.0],
            ),
            create_top_left_aligned_normalized(
                original_grid_shape=(16, 12),
                grid_shape=(8, 6),
                original_voxel_size=(3.1, 2.7),
                downsampling_factor=[2.0, 2.0],
            ),
            create_top_left_aligned_normalized(
                original_grid_shape=(6, 6),
                grid_shape=(2, 3),
                original_voxel_size=(2.0, 1.0),
                downsampling_factor=[3.0, 2.0],
            ),
        ]
        for centered_coordinate_system, top_left_aligned_coordinate_system in zip(
            centered_coordinate_systems, top_left_aligned_coordinate_systems
        ):
            centered = centered_coordinate_system.grid.generate_values()
            top_left_aligned = top_left_aligned_coordinate_system.grid.generate_values()
            assert_close(centered, top_left_aligned)

    def test_top_left_aligned_normalized_coordinate_system(self) -> None:
        """Test that coordinate system is correct"""
        coordinate_systems = [
            create_top_left_aligned_normalized(
                original_grid_shape=(4, 3),
                grid_shape=(3, 2),
                original_voxel_size=(3.0, 2.0),
                downsampling_factor=[1.0, 1.0],
            ),
            create_top_left_aligned_normalized(
                original_grid_shape=(4, 3),
                grid_shape=(4, 3),
                original_voxel_size=(3.0, 2.0),
                downsampling_factor=[3.0, 2.0],
            ),
        ]
        correct_grids = [
            tensor(
                [
                    [[-3 / 4, -3 / 4], [-1 / 4, -1 / 4], [1 / 4, 1 / 4]],
                    [[-1 / 3, 0], [-1 / 3, 0], [-1 / 3, 0]],
                ]
            ),
            tensor(
                [
                    [
                        [-1 / 4, -1 / 4, -1 / 4],
                        [5 / 4, 5 / 4, 5 / 4],
                        [11 / 4, 11 / 4, 11 / 4],
                        [17 / 4, 17 / 4, 17 / 4],
                    ],
                    [
                        [-1 / 6, 1 / 2, 7 / 6],
                        [-1 / 6, 1 / 2, 7 / 6],
                        [-1 / 6, 1 / 2, 7 / 6],
                        [-1 / 6, 1 / 2, 7 / 6],
                    ],
                ]
            ),
        ]
        for coordinate_system, grid in zip(coordinate_systems, correct_grids):
            generated_grid = coordinate_system.grid.generate_values()
            assert_close(generated_grid, grid[None])

    def test_top_left_aligned_coordinate_system(self) -> None:
        """Test that coordinate system is correct"""
        coordinate_systems = [
            create_top_left_aligned(
                original_grid_shape=(4, 3),
                grid_shape=(3, 2),
                original_voxel_size=(3.0, 2.0),
                downsampling_factor=[1.0, 1.0],
            ),
            create_top_left_aligned(
                original_grid_shape=(4, 3),
                grid_shape=(4, 3),
                original_voxel_size=(3.0, 2.0),
                downsampling_factor=[3.0, 2.0],
            ),
        ]
        correct_grids = [
            tensor(
                [
                    [[1.5, 1.5], [4.5, 4.5], [7.5, 7.5]],
                    [[1.0, 3.0], [1.0, 3.0], [1.0, 3.0]],
                ]
            ),
            tensor(
                [
                    [
                        [4.5, 4.5, 4.5],
                        [13.5, 13.5, 13.5],
                        [22.5, 22.5, 22.5],
                        [31.5, 31.5, 31.5],
                    ],
                    [
                        [2.0, 6.0, 10.0],
                        [2.0, 6.0, 10.0],
                        [2.0, 6.0, 10.0],
                        [2.0, 6.0, 10.0],
                    ],
                ]
            ),
        ]
        for coordinate_system, grid in zip(coordinate_systems, correct_grids):
            generated_grid = coordinate_system.grid.generate_values()
            assert_close(generated_grid, grid[None])
