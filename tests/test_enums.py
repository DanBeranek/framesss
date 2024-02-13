"""Test cases for the enums module."""

import pytest

from framesss.enums import DoF


@pytest.mark.parametrize("direction, expected_index", [("x", 0), ("ry", 4), ("rz", 5)])
def test_get_index_valid_directions(direction: str, expected_index: int) -> None:
    # Test that valid directions return the correct index
    assert DoF.get_index(direction) == expected_index


def test_get_index_invalid_direction_raises_error() -> None:
    # Test that an invalid direction raises a ValueError
    with pytest.raises(ValueError):
        DoF.get_index("foo")


def test_dof() -> None:
    x_dof = DoF.TRANSLATION_X

    assert x_dof.index == 0
    assert x_dof.direction == "x"
