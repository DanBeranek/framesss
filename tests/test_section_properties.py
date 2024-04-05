from numpy.testing import assert_almost_equal

from framesss.pre.material import Material
from framesss.pre.section import RectangularSection

DUMMY_MAT = Material("DUMMY", 1, 1, 1, 1)
B = 100
H = 200

A = B * H

Iy = B * H**3 / 12
Iz = H * B**3 / 12
Ix = Iy + Iz


def test_rectangular_section() -> None:
    section = RectangularSection("FOO", B, H, DUMMY_MAT)

    assert_almost_equal(section.area_x, A)
    assert_almost_equal(section.area_y, A)
    assert_almost_equal(section.area_z, A)
    assert_almost_equal(section.inertia_x, Ix)
    assert_almost_equal(section.inertia_y, Iy)
    assert_almost_equal(section.inertia_z, Iz)
    assert_almost_equal(section.height_y, B)
    assert_almost_equal(section.height_z, H)
