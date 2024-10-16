"""Test mappable tensor module"""

from unittest import TestCase

from torch import float32, tensor

from composable_mapping.mappable_tensor.affine_transformation import (
    AffineTransformation,
)
from composable_mapping.mappable_tensor.grid import GridDefinition
from composable_mapping.util import (
    broadcast_tensors_in_parts,
    broadcast_to_in_parts,
    get_spatial_shape,
)


class GridDefinitionTest(TestCase):
    """Test grid definition"""

    def test_broadcast_to_spatial_shape(self):
        """Test broadcast to spatial shape"""
        grid = GridDefinition(
            spatial_shape=(2, 3),
            affine_transformation=AffineTransformation(
                tensor([[2, 3, 1], [0, 3, 2], [0, 0, 1]], dtype=float32),
            ),
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
            affine_transformation=AffineTransformation(
                tensor([[2, 3, 1, 1], [2, 2, 4, 3], [0, 0, 0, 1]], dtype=float32),
            ),
        )
        grid_2 = GridDefinition(
            spatial_shape=(2, 3),
            affine_transformation=AffineTransformation(
                tensor([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=float32),
            ),
        )
        grid_sum_1 = (grid_1 + grid_2).generate()
        generated_grid_1 = grid_1.generate()
        generated_grid_2 = grid_2.generate()
        generated_grid_1, generated_grid_2 = broadcast_tensors_in_parts(
            generated_grid_1, generated_grid_2, n_channel_dims=1
        )
        grid_sum_2 = generated_grid_1 + generated_grid_2
        self.assertTrue(grid_sum_1.allclose(grid_sum_2))
