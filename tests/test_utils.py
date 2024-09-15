"""Test cases for the utils module."""

import numpy as np
from numpy.testing import assert_array_equal

from framesss.utils import assemble_subarray_at_indices
from framesss.utils import is_invertible


def test_assemble_subarray_at_indices() -> None:
    large_array = np.zeros((4, 4))
    small_array = np.array([[1, 2], [3, 4]])
    indices = [0, 2]
    assemble_subarray_at_indices(large_array, small_array, indices)

    expected_array = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert_array_equal(large_array, expected_array)


def test_is_invertible() -> None:
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)

    b[-1] = b[0] + b[1]  # last row is a linear combination of the first two rows

    assert is_invertible(a)
    assert not is_invertible(b)
