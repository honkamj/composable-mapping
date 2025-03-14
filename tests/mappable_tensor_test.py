"""Test mappable tensor module"""

from unittest import TestCase

from torch import eye, float32, randn
from torch.testing import assert_close

from composable_mapping.affine_transformation import AffineTransformation
from composable_mapping.mappable_tensor.grid import GridDefinition
from composable_mapping.mappable_tensor.mappable_tensor import MappableTensor
from composable_mapping.util import (
    broadcast_tensors_in_parts,
    broadcast_to_in_parts,
    get_spatial_shape,
)


def _generate_random_affine_transformation(n_rows: int, n_cols: int) -> AffineTransformation:
    matrix = eye(n_rows, n_cols, dtype=float32)
    matrix[:-1] = randn(n_rows - 1, n_cols)
    return AffineTransformation(matrix)


class GridDefinitionTest(TestCase):
    """Test grid definition"""

    def test_broadcast_to_spatial_shape(self):
        """Test broadcast to spatial shape"""
        grid = GridDefinition(
            spatial_shape=(2, 3),
            affine_transformation=_generate_random_affine_transformation(3, 3),
        )
        broadcasted_1 = grid.broadcast_to_spatial_shape((4, 2, 3)).generate()
        broadcasted_2 = broadcast_to_in_parts(
            grid.generate(), spatial_shape=(4, 2, 3), n_channel_dims=1
        )
        self.assertTrue(get_spatial_shape(broadcasted_1.shape, n_channel_dims=1) == (4, 2, 3))
        self.assertTrue(broadcasted_1.allclose(broadcasted_2))

    def test_add(self):
        """Test add"""
        grid_1 = GridDefinition(
            spatial_shape=(4, 2, 3),
            affine_transformation=_generate_random_affine_transformation(3, 4),
        )
        grid_2 = GridDefinition(
            spatial_shape=(2, 3),
            affine_transformation=_generate_random_affine_transformation(3, 3),
        )
        grid_sum_1 = (grid_1 + grid_2).generate()
        generated_grid_1 = grid_1.generate()
        generated_grid_2 = grid_2.generate()
        generated_grid_1, generated_grid_2 = broadcast_tensors_in_parts(
            generated_grid_1, generated_grid_2, n_channel_dims=1
        )
        grid_sum_2 = generated_grid_1 + generated_grid_2
        assert_close(grid_sum_1, grid_sum_2)


class MappableTensorTest(TestCase):
    """Test mappable tensor"""

    def test_add(self):
        """Test add"""
        tensor_pairs = [
            (
                MappableTensor(
                    displacements=randn(2, 3, 4, 5, 6),
                    affine_transformation=_generate_random_affine_transformation(4, 4),
                    grid=GridDefinition(
                        spatial_shape=(4, 5, 6),
                        affine_transformation=_generate_random_affine_transformation(4, 4),
                    ),
                ),
                MappableTensor(
                    displacements=randn(2, 3, 4, 5, 6),
                    affine_transformation=_generate_random_affine_transformation(4, 4),
                    grid=GridDefinition(
                        spatial_shape=(4, 5, 6),
                        affine_transformation=_generate_random_affine_transformation(4, 4),
                    ),
                ),
            )
        ]
        for tensor_1, tensor_2 in tensor_pairs:
            tensor_sum_1 = (tensor_1 + tensor_2).generate_values()
            tensor_sum_2 = tensor_1.generate_values() + tensor_2.generate_values()
            assert_close(tensor_sum_1, tensor_sum_2)

    def test_apply_affine_transformation(self):
        """Test apply affine transformation"""
        mappable_tensor = MappableTensor(
            displacements=randn(2, 3, 4, 5, 6),
            affine_transformation=_generate_random_affine_transformation(4, 4),
            grid=GridDefinition(
                spatial_shape=(4, 5, 6),
                affine_transformation=_generate_random_affine_transformation(4, 4),
            ),
        )
        transformation = _generate_random_affine_transformation(4, 4)
        transformed_tensor_1 = mappable_tensor.transform(transformation).generate_values()
        transformed_tensor_2 = transformation(mappable_tensor.generate_values())
        assert_close(transformed_tensor_1, transformed_tensor_2)
