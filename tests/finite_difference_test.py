"""Tests for finite difference derivatives"""

from unittest import TestCase

from torch import eye
from torch.testing import assert_close

from composable_mapping.coordinate_system_factory import create_centered_normalized
from composable_mapping.finite_difference import (
    estimate_spatial_jacobian_matrices_for_mapping,
)
from composable_mapping.mapping_factory import create_composable_identity
from composable_mapping.util import broadcast_tensors_in_parts_around_channel_dims


class FiniteDifferenceTests(TestCase):
    """Tests for finite difference derivate estimation"""

    def test_identity_jacobian_of_identity(self) -> None:
        """Test that deriving identity results in identity"""
        coordinate_system = create_centered_normalized((10, 11, 12), (0.9, 1.0, 1.1))
        matrices = estimate_spatial_jacobian_matrices_for_mapping(
            mapping=create_composable_identity(),
            coordinate_system=coordinate_system,
            central=True,
            other_dims="crop",
        )
        matrices, identity = broadcast_tensors_in_parts_around_channel_dims(
            (matrices, eye(3)), n_channel_dims=2
        )
        assert_close(matrices, identity)
