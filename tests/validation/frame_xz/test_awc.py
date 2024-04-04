# mypy: disable-error-code="assignment"
"""
Test examples from American Wood Council.

The test examples were taken from the following publication:
[1] American Wood Council (AWC). (2007). DESIGN AID No. 6 — BEAM DESIGN FORMULAR WITH SHEAR AND MOMENT DIAGRAMS,
    Washington, DC. Available from: https://awc.org/wp-content/uploads/2021/12/AWC-DA6-BeamFormulas-0710.pdf
"""

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from framesss.fea.models.frame_xz import FrameXZModel
from framesss.pre.material import Material
from framesss.pre.section import Section
from framesss.solvers.linear_static import LinearStaticSolver

P = 1000.0  # Force magnitude

P1 = 750.0  # Force magnitude
P2 = 500.0

w = 1000.0  # Distributed load magnitude

w1 = 500.0  # Distributed load magnitude
w2 = 750.0  # Distributed load magnitude

L = 8.0
E = 210.0e9  # Elastic modulus
I = 2.0e-3  # Second moment of inertia

a1 = 0.6 * L
b1 = 0.4 * L

a2 = 0.3 * L
b2 = 0.3 * L
c2 = 0.4 * L

DUMMY_MAT = Material("foo", E, 1, 1, 1)
DUMMY_SEC = Section("bar", 1, 1, 1, 1, I, 1, 1, 1)


def test_figure_1() -> None:
    """[1, Fig. 1] Simple beam - Uniformly Distributed Load."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load([0, 0, -w, 0, 0, -w], load_case)

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R1 = w * L / 2
    R2 = w * L / 2

    shear_forces_z = w * (L / 2 - x)
    bending_moments_y = w * x / 2 * (L - x)

    # Deflection in [1] is considered positive in negative z-direction
    translations_z = -w * x / (24 * E * I) * (L**3 - 2 * L * x**2 + x**3)

    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([0, w * L**2 / 8])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_2() -> None:
    """[1, Fig. 2] Simple beam - Uniform Load Partially Distributed."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    a, b, c = a2, b2, c2

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load(
        [0, 0, -w, 0, 0, -w],
        load_case,
        x_start=a,
        x_end=a + b,
        coordinate_definition="absolute",
    )

    solver.solve()

    x = member.x_local

    # Expected results
    R1 = w * b / (2 * L) * (2 * c + b)
    R2 = w * b / (2 * L) * (2 * a + b)

    # Expected internal forces
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)

    interval = x <= a
    shear_forces_z[interval] = R1
    bending_moments_y[interval] = R1 * x[interval]

    interval = (x > a) * (x < a + b)
    shear_forces_z[interval] = R1 - w * (x[interval] - a)
    bending_moments_y[interval] = R1 * x[interval] - w / 2 * (x[interval] - a) ** 2

    interval = x >= a + b
    shear_forces_z[interval] = -R2
    bending_moments_y[interval] = R2 * (L - x[interval])

    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([0, R1 * (a + R1 / (2 * w))])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_3() -> None:
    """[1, Fig. 3] Simple beam - Uniform Load Partially Distributed at One End."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    a = a1

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load(
        [0, 0, -w, 0, 0, -w],
        load_case,
        x_start=0,
        x_end=a,
        coordinate_definition="absolute",
    )

    solver.solve()

    x = member.x_local

    # Expected results
    R1 = w * a / (2 * L) * (2 * L - a)
    R2 = w * a**2 / (2 * L)

    # Expected internal forces
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)
    translations_z = np.zeros(x.shape)

    interval = x <= a
    shear_forces_z[interval] = R1 - w * x[interval]
    bending_moments_y[interval] = R1 * x[interval] - w * x[interval] ** 2 / 2
    translations_z[interval] = (
        -w
        * x[interval]
        / (24 * E * I * L)
        * (
            a**2 * (2 * L - a) ** 2
            - 2 * a * x[interval] ** 2 * (2 * L - a)
            + L * x[interval] ** 3
        )
    )

    interval = x > a
    shear_forces_z[interval] = np.full(x[interval].shape, -R2)
    bending_moments_y[interval] = R2 * (L - x[interval])
    translations_z[interval] = (
        -w
        * a**2
        * (L - x[interval])
        / (24 * E * I * L)
        * (4 * x[interval] * L - 2 * x[interval] ** 2 - a**2)
    )

    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([0, R1**2 / (2 * w)])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_4() -> None:
    """[1, Fig. 4] Simple beam - Uniform Load Partially Distributed at Each End."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    a, b, c = a2, b2, c2

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load(
        [0, 0, -w1, 0, 0, -w1], load_case, x_end=a, coordinate_definition="absolute"
    )
    member.add_distributed_load(
        [0, 0, -w2, 0, 0, -w2],
        load_case,
        x_start=a + b,
        x_end=L,
        coordinate_definition="absolute",
    )

    solver.solve()

    x = member.x_local

    # Expected results
    R1 = (w1 * a * (2 * L - a) + w2 * c**2) / (2 * L)
    R2 = (w2 * c * (2 * L - c) + w1 * a**2) / (2 * L)

    # Expected internal forces
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)

    interval = x <= a
    shear_forces_z[interval] = R1 - w1 * x[interval]
    bending_moments_y[interval] = R1 * x[interval] - w1 * x[interval] ** 2 / 2

    interval = (x > a) * (x < a + b)
    shear_forces_z[interval] = np.full(x[interval].shape, R1 - w1 * a)
    bending_moments_y[interval] = R1 * x[interval] - w1 * a * (2 * x[interval] - a) / 2

    interval = x >= a + b
    shear_forces_z[interval] = -R2 + w2 * (L - x[interval])
    bending_moments_y[interval] = (
        R2 * (L - x[interval]) - w2 * (L - x[interval]) ** 2 / 2
    )

    min_max_shear = np.sort([R1, -R2])
    if abs(R1) < abs(w1 * a):
        min_max_moment = np.sort([0, R1**2 / (2 * w1)])
    else:
        min_max_moment = np.sort([0, R2**2 / (2 * w2)])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_5() -> None:
    """[1, Fig. 5] Simple beam - Load Increasing Uniformly to One End."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load([0, 0, 0, 0, 0, -w], load_case)

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    W = w * L / 2

    # Expected results
    R1 = W / 3
    R2 = 2 * W / 3

    shear_forces_z = W / 3 - W * x**2 / L**2
    bending_moments_y = W * x * (L**2 - x**2) / (3 * L**2)

    # Deflection in [1] is considered positive in negative z-direction
    translations_z = (
        -W * x * (3 * x**4 - 10 * L**2 * x**2 + 7 * L**4) / (180 * E * I * L**2)
    )

    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([0, 2 * W * L / (9 * np.sqrt(3))])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_6() -> None:
    """[1, Fig. 6] Simple beam - Load Increasing Uniformly to Center."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load([0, 0, 0, 0, 0, -w], load_case, x_end=0.5)
    member.add_distributed_load([0, 0, -w, 0, 0, 0], load_case, x_start=0.5)

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    W = w * L / 2

    # Expected results
    R1 = W / 2
    R2 = W / 2

    # Expected internal forces
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)
    translations_z = np.zeros(x.shape)

    interval = x <= L / 2
    shear_forces_z[interval] = W * (L**2 - 4 * x[interval] ** 2) / (2 * L**2)
    bending_moments_y[interval] = (
        W * x[interval] * (1 / 2 - 2 * x[interval] ** 2 / (3 * L**2))
    )
    translations_z[interval] = (
        -W * x[interval] * (5 * L**2 - 4 * x[interval] ** 2) ** 2 / (480 * E * I * L**2)
    )

    interval = x > L / 2
    shear_forces_z[interval] = -W * (L**2 - 4 * (L - x[interval]) ** 2) / (2 * L**2)
    bending_moments_y[interval] = (
        W * (L - x[interval]) * (1 / 2 - 2 * (L - x[interval]) ** 2 / (3 * L**2))
    )
    translations_z[interval] = (
        -W
        * (L - x[interval])
        * (5 * L**2 - 4 * (L - x[interval]) ** 2) ** 2
        / (480 * E * I * L**2)
    )

    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([0, W * L / 6])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_7() -> None:
    """[1, Fig. 7] Simple beam - Concentrated Load at Center."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    # Load in [1] is applied in the negative z-direction
    member.add_point_load([0, 0, -P, 0, 0, 0], load_case, x=0.5)

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R1 = P / 2
    R2 = P / 2

    # Expected internal forces & displacements
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)
    translations_z = np.zeros(x.shape)

    interval = x <= L / 2
    shear_forces_z[interval] = np.full(x[interval].shape, R1)
    bending_moments_y[interval] = P * x[interval] / 2
    translations_z[interval] = (
        -P * x[interval] * (3 * L**2 - 4 * x[interval] ** 2) / (48 * E * I)
    )

    interval = x > L / 2
    shear_forces_z[interval] = np.full(x[interval].shape, -R2)
    bending_moments_y[interval] = P * (L - x[interval]) / 2
    translations_z[interval] = (
        -P * (L - x[interval]) * (3 * L**2 - 4 * (L - x[interval]) ** 2) / (48 * E * I)
    )

    # Handle the mid-values of shear force
    interval = np.where(np.isclose(x, L / 2))[0]
    shear_forces_z[interval[0]] = R1
    shear_forces_z[interval[1]] = -R2

    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([0, P * L / 4])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_8() -> None:
    """[1, Fig. 8] Simple beam - Concentrated Load at Any Point."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    a, b = a1, b1

    # Load in [1] is applied in the negative z-direction
    member.add_point_load(
        [0, 0, -P, 0, 0, 0], load_case, x=a, coordinate_definition="absolute"
    )

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R1 = P * b / L
    R2 = P * a / L

    # Expected internal forces & displacements
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)
    translations_z = np.zeros(x.shape)

    interval = x <= a
    shear_forces_z[interval] = np.full(x[interval].shape, R1)
    bending_moments_y[interval] = R1 * x[interval]
    translations_z[interval] = (
        -P * b * x[interval] * (L**2 - b**2 - x[interval] ** 2) / (6 * E * I * L)
    )

    interval = x > a
    shear_forces_z[interval] = np.full(x[interval].shape, -R2)
    bending_moments_y[interval] = R2 * (L - x[interval])
    translations_z[interval] = (
        -P
        * a
        * (L - x[interval])
        * (2 * L * x[interval] - x[interval] ** 2 - a**2)
        / (6 * E * I * L)
    )

    # Handle the mid-values of shear force
    interval = np.where(np.isclose(x, a))[0]
    shear_forces_z[interval[0]] = R1
    shear_forces_z[interval[1]] = -R2

    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([0, P * a * b / L])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_9() -> None:
    """[1, Fig. 9] Simple beam - Two Equal Concentrated Loads Symetrically Placed."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    a = 0.3

    # Load in [1] is applied in the negative z-direction
    member.add_point_load(
        [0, 0, -P, 0, 0, 0], load_case, x=a, coordinate_definition="absolute"
    )
    member.add_point_load(
        [0, 0, -P, 0, 0, 0], load_case, x=L - a, coordinate_definition="absolute"
    )

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R1 = P
    R2 = P

    # Expected internal forces & displacements
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)
    translations_z = np.zeros(x.shape)

    interval = x <= a
    shear_forces_z[interval] = np.full(x[interval].shape, R1)
    bending_moments_y[interval] = R1 * x[interval]
    translations_z[interval] = (
        -P * x[interval] * (3 * L * a - 3 * a**2 - x[interval] ** 2) / (6 * E * I)
    )

    interval = np.where(np.isclose(x, a))[0]
    shear_forces_z[interval[0]] = R1
    shear_forces_z[interval[1]] = 0

    interval = (x > a) * (x < L - a)
    shear_forces_z[interval] = np.full(x[interval].shape, 0.0)
    bending_moments_y[interval] = np.full(x[interval].shape, P * a)
    translations_z[interval] = (
        -P * a * (3 * L * x[interval] - 3 * x[interval] ** 2 - a**2) / (6 * E * I)
    )

    interval = x >= L - a
    shear_forces_z[interval] = np.full(x[interval].shape, -R2)
    bending_moments_y[interval] = R2 * (L - x[interval])
    translations_z[interval] = (
        -P
        * (L - x[interval])
        * (3 * L * a - 3 * a**2 - (L - x[interval]) ** 2)
        / (6 * E * I)
    )

    interval = np.where(np.isclose(x, L - a))[0]
    shear_forces_z[interval[0]] = 0
    shear_forces_z[interval[1]] = -R2

    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([0, P * a])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_10() -> None:
    """[1, Fig. 10] Simple Beam - Two Equal Concentrated Loads Unsymmetrically Placed."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    a, b = a2, b2

    # Load in [1] is applied in the negative z-direction
    member.add_point_load(
        [0, 0, -P, 0, 0, 0], load_case, x=a, coordinate_definition="absolute"
    )
    member.add_point_load(
        [0, 0, -P, 0, 0, 0], load_case, x=L - b, coordinate_definition="absolute"
    )

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R1 = P * (L - a + b) / L
    R2 = P * (L - b + a) / L

    # Expected internal forces & displacements
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)

    interval = x <= a
    shear_forces_z[interval] = np.full(x[interval].shape, R1)
    bending_moments_y[interval] = R1 * x[interval]

    interval = (x >= a) * (x <= L - b)
    shear_forces_z[interval] = np.full(x[interval].shape, R1 - P)
    bending_moments_y[interval] = R1 * x[interval] - P * (x[interval] - a)

    interval = x >= L - b
    shear_forces_z[interval] = np.full(x[interval].shape, -R2)
    bending_moments_y[interval] = R2 * (L - x[interval])

    interval = np.where(np.isclose(x, a))[0]
    shear_forces_z[interval[0]] = R1
    shear_forces_z[interval[1]] = R1 - P

    interval = np.where(np.isclose(x, L - b))[0]
    shear_forces_z[interval[0]] = R1 - P
    shear_forces_z[interval[1]] = -R2

    min_max_shear = np.sort([R1, -R2])
    if a > b:
        min_max_moment = np.sort([0, R1 * a])
    else:
        min_max_moment = np.sort([0, R2 * b])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_11() -> None:
    """[1, Fig. 11] Simple Beam - Two Unequal Concentrated Loads Unsymmetrically Placed."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    a, b = a2, b2

    # Load in [1] is applied in the negative z-direction
    member.add_point_load(
        [0, 0, -P1, 0, 0, 0], load_case, x=a, coordinate_definition="absolute"
    )
    member.add_point_load(
        [0, 0, -P2, 0, 0, 0], load_case, x=L - b, coordinate_definition="absolute"
    )

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R1 = (P1 * (L - a) + P2 * b) / L
    R2 = (P1 * a + P2 * (L - b)) / L

    # Expected internal forces & displacements
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)

    interval = x <= a
    shear_forces_z[interval] = np.full(x[interval].shape, R1)
    bending_moments_y[interval] = R1 * x[interval]

    interval = (x >= a) * (x <= L - b)
    shear_forces_z[interval] = np.full(x[interval].shape, R1 - P1)
    bending_moments_y[interval] = R1 * x[interval] - P1 * (x[interval] - a)

    interval = x >= L - b
    shear_forces_z[interval] = np.full(x[interval].shape, -R2)
    bending_moments_y[interval] = R2 * (L - x[interval])

    interval = np.where(np.isclose(x, a))[0]
    shear_forces_z[interval[0]] = R1
    shear_forces_z[interval[1]] = R1 - P1

    interval = np.where(np.isclose(x, L - b))[0]
    shear_forces_z[interval[0]] = R1 - P1
    shear_forces_z[interval[1]] = -R2

    min_max_shear = np.sort([R1, -R2])
    if abs(R1) < abs(P1):
        min_max_moment = np.sort([0, R1 * a])
    else:
        min_max_moment = np.sort([0, R2 * b])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_11_combinations() -> None:
    """[1, Fig. 11] Simple Beam - Two Unequal Concentrated Loads Unsymmetrically Placed."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    lc1 = model.add_load_case("LC1")
    lc2 = model.add_load_case("LC2")

    combination = {
        lc1: 1.0,
        lc2: 2.0,
    }

    comb = model.add_load_case_combination("CO1", combination)

    solver = LinearStaticSolver(model)

    a, b = a2, b2

    # Load in [1] is applied in the negative z-direction
    member.add_point_load(
        [0, 0, -P1, 0, 0, 0], lc1, x=a, coordinate_definition="absolute"
    )
    member.add_point_load(
        [0, 0, -P2 / 2, 0, 0, 0], lc2, x=L - b, coordinate_definition="absolute"
    )

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R1 = (P1 * (L - a) + P2 * b) / L
    R2 = (P1 * a + P2 * (L - b)) / L

    # Expected internal forces & displacements
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)

    interval = x <= a
    shear_forces_z[interval] = np.full(x[interval].shape, R1)
    bending_moments_y[interval] = R1 * x[interval]

    interval = (x >= a) * (x <= L - b)
    shear_forces_z[interval] = np.full(x[interval].shape, R1 - P1)
    bending_moments_y[interval] = R1 * x[interval] - P1 * (x[interval] - a)

    interval = x >= L - b
    shear_forces_z[interval] = np.full(x[interval].shape, -R2)
    bending_moments_y[interval] = R2 * (L - x[interval])

    interval = np.where(np.isclose(x, a))[0]
    shear_forces_z[interval[0]] = R1
    shear_forces_z[interval[1]] = R1 - P1

    interval = np.where(np.isclose(x, L - b))[0]
    shear_forces_z[interval[0]] = R1 - P1
    shear_forces_z[interval[1]] = -R2

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[comb], R1)
    assert_almost_equal(node_2.results.reaction_force_z[comb], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[comb], shear_forces_z)
    assert_array_almost_equal(member.results.bending_moments_y[comb], bending_moments_y)


def test_figure_12() -> None:
    """[1, Fig. 12] Cantilever Beam - Uniformly Distributed Load."""
    model = FrameXZModel()
    node_1 = model.add_node("N1", [], ["free", "free", "free", "free", "free", "free"])
    node_2 = model.add_node(
        "N2", [L], ["fixed", "free", "fixed", "free", "fixed", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load([0, 0, -w, 0, 0, -w], load_case)

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R = w * L
    M = w * L**2 / 2

    # Expected internal forces & displacements
    shear_forces_z = -w * x
    bending_moments_y = -w * x**2 / 2
    translations_z = -w * (x**4 - 4 * L**3 * x + 3 * L**4) / (24 * E * I)

    min_max_shear = np.sort([0, -R])
    min_max_moment = np.sort([0, -M])

    # Reactions
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R)
    assert_almost_equal(node_2.results.reaction_moment_y[load_case], M)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_13() -> None:
    """[1, Fig. 13] Cantilever Beam - Concentrated Load at Free End."""
    model = FrameXZModel()
    node_1 = model.add_node("N1", [], ["free", "free", "free", "free", "free", "free"])
    node_2 = model.add_node(
        "N2", [L], ["fixed", "free", "fixed", "free", "fixed", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    # Load in [1] is applied in the negative z-direction
    node_1.add_nodal_load([0, 0, -P, 0, 0, 0], load_case)

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R = P
    M = P * L

    # Expected internal forces & displacements
    shear_forces_z = np.full(x.shape, -R)
    bending_moments_y = -P * x
    translations_z = -P * (2 * L**3 - 3 * L**2 * x + x**3) / (6 * E * I)

    min_max_shear = np.sort([-R, -R])
    min_max_moment = np.sort([0, -M])

    # Reactions
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R)
    assert_almost_equal(node_2.results.reaction_moment_y[load_case], M)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_14() -> None:
    """[1, Fig. 14] Cantilever Beam - Concentrated Load at Any Point."""
    model = FrameXZModel()
    node_1 = model.add_node("N1", [], ["free", "free", "free", "free", "free", "free"])
    node_2 = model.add_node(
        "N2", [L], ["fixed", "free", "fixed", "free", "fixed", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    a, b = a1, b1

    # Load in [1] is applied in the negative z-direction
    member.add_point_load(
        [0, 0, -P, 0, 0, 0], load_case, x=a, coordinate_definition="absolute"
    )

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R = P
    M = P * b

    # Expected internal forces & displacements
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)
    translations_z = np.zeros(x.shape)

    interval = x <= a
    shear_forces_z[interval] = np.full(x[interval].shape, 0)
    bending_moments_y[interval] = np.full(x[interval].shape, 0)
    translations_z[interval] = -P * b**2 * (3 * L - 3 * x[interval] - b) / (6 * E * I)

    interval = x > a
    shear_forces_z[interval] = np.full(x[interval].shape, -R)
    bending_moments_y[interval] = -P * (x[interval] - a)
    translations_z[interval] = (
        -P * (L - x[interval]) ** 2 * (3 * b - L + x[interval]) / (6 * E * I)
    )

    interval = np.where(np.isclose(x, a))[0]
    shear_forces_z[interval[0]] = 0
    shear_forces_z[interval[1]] = -R

    min_max_shear = np.sort([0, -R])
    min_max_moment = np.sort([0, -M])

    # Reactions
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R)
    assert_almost_equal(node_2.results.reaction_moment_y[load_case], M)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_15() -> None:
    """[1, Fig. 15] Beam Fixed at One End, Supported at Other - Uniformly Distributed Load."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["fixed", "free", "fixed", "free", "fixed", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load([0, 0, -w, 0, 0, -w], load_case)

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R1 = 3 * w * L / 8
    R2 = 5 * w * L / 8
    M = w * L**2 / 8

    # Expected internal forces & displacements
    shear_forces_z = R1 - w * x
    bending_moments_y = R1 * x - w * x**2 / 2
    translations_z = -w * x * (L**3 - 3 * L * x**2 + 2 * x**3) / (48 * E * I)
    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([9 * w * L**2 / 128, -M])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)
    assert_almost_equal(node_2.results.reaction_moment_y[load_case], M)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_16() -> None:
    """[1, Fig. 16] Beam Fixed at One End, Supported at Other - Concentrated Load at Center."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["fixed", "free", "fixed", "free", "fixed", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    # Load in [1] is applied in the negative z-direction
    member.add_point_load([0, 0, -P, 0, 0, 0], load_case, x=0.5)

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R1 = 5 * P / 16
    R2 = 11 * P / 16
    M = 3 * P * L / 16

    # Expected internal forces & displacements
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)
    translations_z = np.zeros(x.shape)

    interval = x <= L / 2
    shear_forces_z[interval] = R1
    bending_moments_y[interval] = 5 * P * x[interval] / 16
    translations_z[interval] = (
        -P * x[interval] * (3 * L**2 - 5 * x[interval] ** 2) / (96 * E * I)
    )

    interval = x > L / 2
    shear_forces_z[interval] = -R2
    bending_moments_y[interval] = P * (L / 2 - 11 * x[interval] / 16)
    translations_z[interval] = (
        -P * (x[interval] - L) ** 2 * (11 * x[interval] - 2 * L) / (96 * E * I)
    )

    interval = np.where(np.isclose(x, L / 2))[0]
    shear_forces_z[interval[0]] = R1
    shear_forces_z[interval[1]] = -R2

    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([5 * P * L / 32, -M])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)
    assert_almost_equal(node_2.results.reaction_moment_y[load_case], M)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_17() -> None:
    """[1, Fig. 17] Beam Fixed at One End, Supported at Other - Concentrated Load at Any Point."""
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["fixed", "free", "fixed", "free", "fixed", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    a, b = a1, b1

    # Load in [1] is applied in the negative z-direction
    member.add_point_load(
        [0, 0, -P, 0, 0, 0], load_case, x=a, coordinate_definition="absolute"
    )

    solver = LinearStaticSolver(model)
    solver.solve()

    x = member.x_local

    # Expected results
    R1 = P * b**2 * (a + 2 * L) / (2 * L**3)
    R2 = P * a * (3 * L**2 - a**2) / (2 * L**3)
    M = P * a * b * (a + L) / (2 * L**2)

    # Expected internal forces & displacements
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)
    translations_z = np.zeros(x.shape)

    interval = x <= a
    shear_forces_z[interval] = R1
    bending_moments_y[interval] = R1 * x[interval]
    translations_z[interval] = (
        -P
        * b**2
        * x[interval]
        * (3 * a * L**2 - 2 * L * x[interval] ** 2 - a * x[interval] ** 2)
        / (12 * E * I * L**3)
    )

    interval = x > a
    shear_forces_z[interval] = -R2
    bending_moments_y[interval] = R1 * x[interval] - P * (x[interval] - a)
    translations_z[interval] = (
        -P
        * a
        * (L - x[interval]) ** 2
        * (3 * L**2 * x[interval] - a**2 * x[interval] - 2 * a**2 * L)
        / (12 * E * I * L**3)
    )

    interval = np.where(np.isclose(x, a))[0]
    shear_forces_z[interval[0]] = R1
    shear_forces_z[interval[1]] = -R2

    min_max_shear = np.sort([R1, -R2])
    min_max_moment = np.sort([R1 * a, -M])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)
    assert_almost_equal(node_2.results.reaction_moment_y[load_case], M)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_18() -> None:
    """
    [1, Fig. 18] Beam Overhanging One Support - Uniform Distributed Load.

    Note: The beam is modeled as one :class:`Member1D`.
    """
    a = 1

    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_3 = model.add_node(
        "N3", [L + a], ["free", "free", "free", "free", "free", "free"]
    )
    member = model.add_member("M1", "navier", [node_1, node_3], DUMMY_MAT, DUMMY_SEC)

    node_2 = member.add_node(
        "N2",
        L,
        ["free", "free", "fixed", "free", "free", "free"],
        coordinate_definition="absolute",
    )

    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    # Load in [1] is applied in the negative z-direction
    member.add_distributed_load([0, 0, -w, 0, 0, -w], load_case)

    solver.solve(True)

    x = member.x_local

    # Expected results
    V1 = w * (L**2 - a**2) / (2 * L)
    V2 = w * a
    V3 = w * (L**2 + a**2) / (2 * L)

    R1 = 1 * V1
    R2 = V2 + V3

    # Expected internal forces & displacements
    shear_forces_z = np.zeros(x.shape)
    bending_moments_y = np.zeros(x.shape)
    translations_z = np.zeros(x.shape)

    interval = x <= L
    shear_forces_z[interval] = R1 - w * x[interval]
    bending_moments_y[interval] = (
        w * x[interval] * (L**2 - a**2 - x[interval] * L) / (2 * L)
    )
    translations_z[interval] = (
        -w
        * x[interval]
        * (
            L**4
            - 2 * L**2 * x[interval] ** 2
            + L * x[interval] ** 3
            - 2 * a**2 * L**2
            + 2 * a**2 * x[interval] ** 2
        )
        / (24 * E * I * L)
    )

    interval = x > L
    xl = x[interval] - L
    shear_forces_z[interval] = w * (a - xl)
    bending_moments_y[interval] = -w * (a - xl) ** 2 / 2
    translations_z[interval] = (
        -w
        * xl
        * (4 * a**2 * L - L**3 + 6 * a**2 * xl - 4 * a * xl**2 + xl**3)
        / (24 * E * I)
    )

    interval = np.where(np.isclose(x, L))[0]
    shear_forces_z[interval[0]] = -V3
    shear_forces_z[interval[1]] = V2

    min_max_shear = np.sort([np.sign(V2) * np.max([abs(V1), abs(V2)]), -V3])
    min_max_moment = np.sort(
        [w * (L + a) ** 2 * (L - a) ** 2 / (8 * L**2), -w * a**2 / 2]
    )

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(member.results.shear_forces_z[load_case], shear_forces_z)
    assert_array_almost_equal(
        member.results.bending_moments_y[load_case], bending_moments_y
    )

    # Displacements
    assert_array_almost_equal(member.results.translations_z[load_case], translations_z)

    # Extremes
    assert_array_almost_equal(
        member.results.min_max_shear_forces_z[load_case], min_max_shear
    )
    assert_array_almost_equal(
        member.results.min_max_bending_moments_y[load_case], min_max_moment
    )


def test_figure_19() -> None:
    """
    [1, Fig. 19] Beam Overhanging One Support - Uniformly Distributed Load on Overhang.

    Note: Beam is modeled as 2 :class:`Member1D`.
    Index 1: Beam between supports,
    Index 2: Overhang
    """
    a = 1

    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [L], ["free", "free", "fixed", "free", "free", "free"]
    )
    node_3 = model.add_node(
        "N3", [L + a], ["free", "free", "free", "free", "free", "free"]
    )

    member_1 = model.add_member("M1", "navier", [node_1, node_2], DUMMY_MAT, DUMMY_SEC)
    member_2 = model.add_member("M2", "navier", [node_2, node_3], DUMMY_MAT, DUMMY_SEC)

    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    # Load in [1] is applied in the negative z-direction
    member_2.add_distributed_load([0, 0, -w, 0, 0, -w], load_case)

    solver.solve()

    x = member_1.x_local
    x1 = member_2.x_local

    # Expected reactions
    R1 = -w * a**2 / (2 * L)
    R2 = w * a * (2 * L + a) / (2 * L)

    # Expected internal forces & displacements
    shear_forces_z_1 = np.full(x.shape, R1)
    shear_forces_z_2 = w * (a - x1)

    bending_moments_y_1 = R1 * x
    bending_moments_y_2 = -w * (a - x1) ** 2 / 2

    translations_z_1 = w * a**2 * x * (L**2 - x**2) / (12 * E * I * L)
    translations_z_2 = (
        -w * x1 * (4 * a**2 * L + 6 * a**2 * x1 - 4 * a * x1**2 + x1**3) / (24 * E * I)
    )

    min_max_shear_1 = np.sort([R1, R1])
    min_max_shear_2 = np.sort([0, w * a])
    min_max_moment_1 = np.sort([0, -w * a**2 / 2])
    min_max_moment_2 = np.sort([0, -w * a**2 / 2])

    # Reactions
    assert_almost_equal(node_1.results.reaction_force_z[load_case], R1)
    assert_almost_equal(node_2.results.reaction_force_z[load_case], R2)

    # Internal forces
    assert_array_almost_equal(
        member_1.results.shear_forces_z[load_case], shear_forces_z_1
    )
    assert_array_almost_equal(
        member_1.results.bending_moments_y[load_case], bending_moments_y_1
    )

    assert_array_almost_equal(
        member_2.results.shear_forces_z[load_case], shear_forces_z_2
    )
    assert_array_almost_equal(
        member_2.results.bending_moments_y[load_case], bending_moments_y_2
    )

    # Displacements
    assert_array_almost_equal(
        member_1.results.translations_z[load_case], translations_z_1
    )
    assert_array_almost_equal(
        member_2.results.translations_z[load_case], translations_z_2
    )

    # Extremes
    assert_array_almost_equal(
        member_1.results.min_max_shear_forces_z[load_case], min_max_shear_1
    )
    assert_array_almost_equal(
        member_1.results.min_max_bending_moments_y[load_case], min_max_moment_1
    )

    assert_array_almost_equal(
        member_2.results.min_max_shear_forces_z[load_case], min_max_shear_2
    )
    assert_array_almost_equal(
        member_2.results.min_max_bending_moments_y[load_case], min_max_moment_2
    )
