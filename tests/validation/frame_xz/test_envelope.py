import numpy as np
from numpy.testing import assert_array_almost_equal

from framesss.fea.models.frame_xz import FrameXZModel
from framesss.pre.material import Material
from framesss.pre.section import Section
from framesss.solvers.linear_static import LinearStaticSolver

L = 4.0  # Length
F = 2  # Force
E = 210.0e9  # Elastic modulus
I = 2.0e-3  # Second moment of inertia
DUMMY_MAT = Material("foo", E, 1, 1, 1)
DUMMY_SEC = Section("bar", 1, 1, 1, 1, I, 1, 1, 1, DUMMY_MAT)


def test_simple_beam_3_point_loads() -> None:
    """Simple beam - Uniformly Distributed Load."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_SEC)

    lc1 = model.add_load_case("LC1")
    member.add_point_load(
        [0, 0, -F, 0, 0, 0], lc1, x=L / 4, coordinate_definition="absolute"
    )

    lc2 = model.add_load_case("LC2")
    member.add_point_load(
        [0, 0, +F, 0, 0, 0], lc2, x=L / 2, coordinate_definition="absolute"
    )

    lc3 = model.add_load_case("LC3")
    member.add_point_load(
        [0, 0, -F, 0, 0, 0], lc3, x=3 * L / 4, coordinate_definition="absolute"
    )

    envelope = model.add_envelope("ENV1", [lc1, lc2, lc3])

    solver = LinearStaticSolver(model)
    solver.solve(verbose=True)

    # EXPECTED RESULTS:
    x = member.x_local

    min_moments_1 = np.linspace(0, -1, int(x.shape[0] / 4))
    min_moments_2 = np.linspace(-1, -2, int(x.shape[0] / 4))
    min_moments = np.concatenate((min_moments_1, min_moments_2))
    min_moments = np.concatenate((min_moments, np.flip(min_moments)))

    max_moments_1 = np.linspace(0, 1.5, int(x.shape[0] / 4))
    max_moments_2 = np.linspace(1.5, 1, int(x.shape[0] / 4))
    max_moments = np.concatenate((max_moments_1, max_moments_2))
    max_moments = np.concatenate((max_moments, np.flip(max_moments)))

    assert_array_almost_equal(
        member.results.bending_moments_y[envelope],
        np.array([min_moments, max_moments]),
    )

    min_shear_1 = np.full(int(x.shape[0] / 4), -1.0)
    min_shear_2 = np.full(int(x.shape[0] / 4), -1.0)
    min_shear_3 = np.full(int(x.shape[0] / 4), -0.5)
    min_shear_4 = np.full(int(x.shape[0] / 4), -1.5)

    min_shear = np.concatenate((min_shear_1, min_shear_2, min_shear_3, min_shear_4))

    max_shear_1 = np.full(int(x.shape[0] / 4), +1.5)
    max_shear_2 = np.full(int(x.shape[0] / 4), +0.5)
    max_shear_3 = np.full(int(x.shape[0] / 4), +1.0)
    max_shear_4 = np.full(int(x.shape[0] / 4), +1.0)

    max_shear = np.concatenate((max_shear_1, max_shear_2, max_shear_3, max_shear_4))

    assert_array_almost_equal(
        member.results.shear_forces_z[envelope], np.array([min_shear, max_shear])
    )
