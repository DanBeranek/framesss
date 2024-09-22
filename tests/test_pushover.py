import numpy as np

from framesss.fea.models.frame_xz import FrameXZModel
from framesss.pre.material import Material
from framesss.pre.section import Section
from framesss.solvers.linear_static import LinearStaticSolver
from framesss.solvers.pushover import PushoverSolver

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

DUMMY_MAT = Material(
    label="foo",
    elastic_modulus=30e9,
    poissons_ratio=0.2,
    thermal_expansion_coefficient=10e-7,
    density=2300)
DUMMY_SEC = Section(
    label="bar",
    area_x=0.18,
    area_y=0.18,
    area_z=0.18,
    inertia_x=6.75e-3,
    inertia_y=31_250_000/30e9,
    inertia_z=1.35e-3,
    height_y=0.3,
    height_z=0.6,
    material=DUMMY_MAT,
    moment_curvature=np.array([
        [300e3, 250e3, -250e3, -300e3],
        [-0.06, -0.008, 0.008, 0.06]
    ])
)


def test_linear() -> None:
    """
    Simple beam under uniformly distributed load 80 kN/m.

    Max bending moment = 250 kNm
    Result from pushover and from linear should be the same,
    because its in the linear interval.
    """
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [0, 0, 0], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [5, 0, 0], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_SEC)
    lc1 = model.add_load_case("LC1")

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load([0, 0, -30000, 0, 0, -30000], lc1)

    lc2 = model.add_load_case("LC2")

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load([0, 0, -50000, 0, 0, -50000], lc2)

    nlc = model.add_nonlinear_load_case_combination(
        label="NLC1",
        combination={
            lc1: 1.0,
            lc2: 1.0
        }
    )

    solver = PushoverSolver(model)
    solver.solve(verbose=True, modulus_type='secant')

    pushover_max_moment = np.max(member.results.bending_moments_y[nlc])
    pushover_max_deflection = np.min(member.results.translations_z[nlc])

    model_linear = FrameXZModel()
    node_1 = model_linear.add_node(
        "N1", [0, 0, 0], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model_linear.add_node(
        "N2", [5, 0, 0], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model_linear.add_member("M1", "navier", [node_1, node_2], DUMMY_SEC)
    lc1 = model_linear.add_load_case("LC1")

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load([0, 0, -80000, 0, 0, -80000], lc1)

    solver = LinearStaticSolver(model_linear)
    solver.solve()

    linear_max_moment = np.max(member.results.bending_moments_y[lc1])
    linear_max_deflection = np.min(member.results.translations_z[lc1])

    assert_almost_equal(linear_max_moment, pushover_max_moment, decimal=5)
    assert_almost_equal(linear_max_deflection, pushover_max_deflection)
