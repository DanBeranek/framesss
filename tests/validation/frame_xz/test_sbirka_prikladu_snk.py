"""
Test examples from Jíra.

Test examples were taken from the following publication:
[2] Jíra, A. a kol.: Sbírka příkladů stavební mechaniky: princip virtuálních sil, silová metoda, deformační metoda,
    ČVUT v Praze, 2019, ISBN: 978-80-01-06677-5, available from:
    https://mech.fsv.cvut.cz/wiki/images/9/9e/Sbirka_prikladu_SNK.pdf
"""

import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from framesss.fea.models.frame_xz import FrameXZModel
from framesss.pre.material import Material
from framesss.pre.section import Section
from framesss.solvers.linear_static import LinearStaticSolver


def test_example_5_2() -> None:
    """
    [2, Ex. 5.2] Frame under force load.

    In the publication [2], the Z-axis points downwards, but in framesss, the Z-axis
    is considered pointing upwards. Because of this, the translations in the Z-axis
    direction taken from [2] are multiplied by -1.
    """
    material = Material("test", 30.0e6, 0.2, 1, 0)
    section = Section("test", 4.0e-3, 4.0e-3, 4.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1, 1)

    model = FrameXZModel()

    node_1 = model.add_node(
        "N1", [0, 0, 0], ["FIXed", "free", "fixed", "free", "fixed", "free"]
    )
    node_2 = model.add_node(
        "N2", [0, 0, 4], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_3 = model.add_node("N3", [3, 0, 4])
    node_4 = model.add_node(
        "N4", [6, 0, 0], ["fixed", "free", "fixed", "free", "fixed", "free"]
    )
    node_5 = model.add_node("N5", [3, 0, 5.5])
    node_6 = model.add_node(
        "N6", [0, 0, 5.5], ["free", "free", "fixed", "free", "free", "free"]
    )

    member_12 = model.add_member("1-2", "navier", [node_1, node_2], material, section)
    member_23 = model.add_member(
        "2-3", "navier", [node_2, node_3], material, section, ["fixed", "hinged"]
    )
    member_34 = model.add_member(
        "3-4", "navier", [node_3, node_4], material, section, ["hinged", "fixed"]
    )
    member_35 = model.add_member(
        "3-5", "navier", [node_3, node_5], material, section, ["hinged", "fixed"]
    )
    member_65 = model.add_member("6-5", "navier", [node_6, node_5], material, section)

    lc = model.add_load_case("LC1")

    member_12.add_distributed_load([10, 0, 0, 10, 0, 0], lc)
    member_34.add_point_load([0, 0, -15, 0, 0, 0], lc, x=1.8 / 3)
    member_65.add_point_load([0, 0, -20, 0, 0, 0], lc, x=0.5)

    node_6.add_nodal_load([4, 0, 0, 0, 0, 0], lc)

    solver = LinearStaticSolver(model)
    solver.solve()

    assert_almost_equal(node_1.results.reaction_force_x[lc], -20.781, decimal=3)
    assert_almost_equal(node_1.results.reaction_force_z[lc], 0, decimal=3)
    assert_almost_equal(node_1.results.reaction_moment_y[lc], -14.375, decimal=3)

    assert_almost_equal(node_2.results.reaction_force_x[lc], -15.258, decimal=3)
    assert_almost_equal(node_2.results.reaction_force_z[lc], 3.75, decimal=3)

    assert_almost_equal(node_2.results.rotation_y[lc], -6.942e-5)

    assert_almost_equal(node_3.results.translation_x[lc], -9.903e-5)
    assert_almost_equal(node_3.results.translation_z[lc], -9.168e-4)

    assert_almost_equal(node_4.results.reaction_force_x[lc], -7.961, decimal=3)
    assert_almost_equal(node_4.results.reaction_force_z[lc], 23.249, decimal=3)
    assert_almost_equal(node_4.results.reaction_moment_y[lc], 10.906, decimal=3)

    assert_almost_equal(node_6.results.reaction_force_z[lc], 8, decimal=3)

    assert_array_almost_equal(
        member_12.results.min_max_axial_forces[lc], [0, 0], decimal=3
    )
    assert_array_almost_equal(
        member_12.results.min_max_shear_forces_z[lc], [-19.219, 20.781], decimal=3
    )
    assert_array_almost_equal(
        member_12.results.min_max_bending_moments_y[lc], [-14.375, 7.217], decimal=3
    )

    assert_array_almost_equal(
        member_23.results.min_max_axial_forces[lc], [-3.961, -3.961], decimal=3
    )
    assert_array_almost_equal(
        member_23.results.min_max_shear_forces_z[lc], [3.75, 3.75], decimal=3
    )
    assert_array_almost_equal(
        member_23.results.min_max_bending_moments_y[lc], [-11.251, 0], decimal=3
    )

    assert_array_almost_equal(
        member_34.results.min_max_axial_forces[lc], [-23.376, -11.376], decimal=3
    )
    assert_array_almost_equal(
        member_34.results.min_max_shear_forces_z[lc], [-7.581, 1.419], decimal=3
    )
    assert_array_almost_equal(
        member_34.results.min_max_bending_moments_y[lc], [-10.906, 4.257], decimal=3
    )

    assert_array_almost_equal(
        member_35.results.min_max_axial_forces[lc], [-12, -12], decimal=3
    )
    assert_array_almost_equal(
        member_35.results.min_max_shear_forces_z[lc], [4, 4], decimal=3
    )
    assert_array_almost_equal(
        member_35.results.min_max_bending_moments_y[lc], [-0, 6], decimal=3
    )

    assert_array_almost_equal(
        member_65.results.min_max_axial_forces[lc], [-4, -4], decimal=3
    )
    assert_array_almost_equal(
        member_65.results.min_max_shear_forces_z[lc], [-12, 8], decimal=3
    )
    assert_array_almost_equal(
        member_65.results.min_max_bending_moments_y[lc], [-6, 12], decimal=3
    )


def test_example_5_2_combinations() -> None:
    """
    [2, Ex. 5.2] Frame under force load.

    In the publication [2], the Z-axis points downwards, but in framesss, the Z-axis
    is considered pointing upwards. Because of this, the translations in the Z-axis
    direction taken from [2] are multiplied by -1.
    """
    material = Material("test", 30.0e6, 0.2, 1, 0)
    section = Section("test", 4.0e-3, 4.0e-3, 4.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1, 1)

    model = FrameXZModel()

    node_1 = model.add_node(
        "N1", [0, 0, 0], ["FIXed", "free", "fixed", "free", "fixed", "free"]
    )
    node_2 = model.add_node(
        "N2", [0, 0, 4], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_3 = model.add_node("N3", [3, 0, 4])
    node_4 = model.add_node(
        "N4", [6, 0, 0], ["fixed", "free", "fixed", "free", "fixed", "free"]
    )
    node_5 = model.add_node("N5", [3, 0, 5.5])
    node_6 = model.add_node(
        "N6", [0, 0, 5.5], ["free", "free", "fixed", "free", "free", "free"]
    )

    member_12 = model.add_member("1-2", "navier", [node_1, node_2], material, section)
    member_23 = model.add_member(
        "2-3", "navier", [node_2, node_3], material, section, ["fixed", "hinged"]
    )
    member_34 = model.add_member(
        "3-4", "navier", [node_3, node_4], material, section, ["hinged", "fixed"]
    )
    member_35 = model.add_member(
        "3-5", "navier", [node_3, node_5], material, section, ["hinged", "fixed"]
    )
    member_65 = model.add_member("6-5", "navier", [node_6, node_5], material, section)

    lc1 = model.add_load_case("LC1")
    f1 = 2
    member_12.add_distributed_load(np.array([10, 0, 0, 10, 0, 0]) / f1, lc1)

    lc2 = model.add_load_case("LC2")
    f2 = -1
    member_34.add_point_load(np.array([0, 0, -15, 0, 0, 0]) / f2, lc2, x=1.8 / 3)

    lc3 = model.add_load_case("LC3")
    f3 = 1
    member_65.add_point_load(np.array([0, 0, -20, 0, 0, 0]) / f3, lc3, x=0.5)

    lc4 = model.add_load_case("LC4")
    f4 = 4
    node_6.add_nodal_load(np.array([4, 0, 0, 0, 0, 0]) / f4, lc4)

    comb = model.add_load_case_combination("CO1", {lc1: f1, lc2: f2, lc3: f3, lc4: f4})

    solver = LinearStaticSolver(model)
    solver.solve()

    assert_almost_equal(node_1.results.reaction_force_x[comb], -20.781, decimal=3)
    assert_almost_equal(node_1.results.reaction_force_z[comb], 0, decimal=3)
    assert_almost_equal(node_1.results.reaction_moment_y[comb], -14.375, decimal=3)

    assert_almost_equal(node_2.results.reaction_force_x[comb], -15.258, decimal=3)
    assert_almost_equal(node_2.results.reaction_force_z[comb], 3.75, decimal=3)

    assert_almost_equal(node_2.results.rotation_y[comb], -6.942e-5)

    assert_almost_equal(node_3.results.translation_x[comb], -9.903e-5)
    assert_almost_equal(node_3.results.translation_z[comb], -9.168e-4)

    assert_almost_equal(node_4.results.reaction_force_x[comb], -7.961, decimal=3)
    assert_almost_equal(node_4.results.reaction_force_z[comb], 23.249, decimal=3)
    assert_almost_equal(node_4.results.reaction_moment_y[comb], 10.906, decimal=3)

    assert_almost_equal(node_6.results.reaction_force_z[comb], 8, decimal=3)

    assert_array_almost_equal(
        member_12.results.min_max_axial_forces[comb], [0, 0], decimal=3
    )
    assert_array_almost_equal(
        member_12.results.min_max_shear_forces_z[comb], [-19.219, 20.781], decimal=3
    )
    assert_array_almost_equal(
        member_12.results.min_max_bending_moments_y[comb], [-14.375, 7.217], decimal=3
    )

    assert_array_almost_equal(
        member_23.results.min_max_axial_forces[comb], [-3.961, -3.961], decimal=3
    )
    assert_array_almost_equal(
        member_23.results.min_max_shear_forces_z[comb], [3.75, 3.75], decimal=3
    )
    assert_array_almost_equal(
        member_23.results.min_max_bending_moments_y[comb], [-11.251, 0], decimal=3
    )

    assert_array_almost_equal(
        member_34.results.min_max_axial_forces[comb], [-23.376, -11.376], decimal=3
    )
    assert_array_almost_equal(
        member_34.results.min_max_shear_forces_z[comb], [-7.581, 1.419], decimal=3
    )
    assert_array_almost_equal(
        member_34.results.min_max_bending_moments_y[comb], [-10.906, 4.257], decimal=3
    )

    assert_array_almost_equal(
        member_35.results.min_max_axial_forces[comb], [-12, -12], decimal=3
    )
    assert_array_almost_equal(
        member_35.results.min_max_shear_forces_z[comb], [4, 4], decimal=3
    )
    assert_array_almost_equal(
        member_35.results.min_max_bending_moments_y[comb], [-0, 6], decimal=3
    )

    assert_array_almost_equal(
        member_65.results.min_max_axial_forces[comb], [-4, -4], decimal=3
    )
    assert_array_almost_equal(
        member_65.results.min_max_shear_forces_z[comb], [-12, 8], decimal=3
    )
    assert_array_almost_equal(
        member_65.results.min_max_bending_moments_y[comb], [-6, 12], decimal=3
    )


def test_example_5_3() -> None:
    """[2, Ex. 5.3] Frame under thermal load with prescribed displacements."""
    material = Material("test", 20.0e3, 0.2, 12.0e-6, 0)
    section = Section("test", 5, 5, 5, 1, 1, 1, 0.3, 0.3)

    model = FrameXZModel()

    node_1 = model.add_node(
        "1", [0, 0, 0], fixity=["fixed", "free", "fixed", "free", "fixed", "free"]
    )
    node_2 = model.add_node("2", [4, 0, 0])
    node_3 = model.add_node(
        "3", [4, 0, -4], fixity=["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_4 = model.add_node(
        "4", [8, 0, -4], fixity=["fixed", "free", "fixed", "free", "fixed", "free"]
    )

    member_12 = model.add_member("1-2", "navier", [node_1, node_2], material, section)
    member_32 = model.add_member(
        "3-2", "navier", [node_3, node_2], material, section, ["fixed", "fixed"]
    )
    member_34 = model.add_member(
        "3-4", "navier", [node_3, node_4], material, section, ["hinged", "fixed"]
    )

    lc = model.add_load_case("LC1")

    member_32.add_thermal_load([15, 0, 50], lc)
    node_1.add_prescribed_displacement(0.005, "z", lc)
    node_4.add_prescribed_displacement(np.deg2rad(10 / 60), "ry", lc)

    solver = LinearStaticSolver(model)
    solver.solve()

    assert_almost_equal(node_1.results.reaction_force_x[lc], -11.131, decimal=3)
    assert_almost_equal(node_1.results.reaction_force_z[lc], 19.960, decimal=3)
    assert_almost_equal(node_1.results.reaction_moment_y[lc], -35.317, decimal=3)

    assert_almost_equal(node_2.results.translation_x[lc], 4.452e-4)
    assert_almost_equal(node_2.results.translation_z[lc], 1.518e-3, decimal=6)
    assert_almost_equal(node_2.results.rotation_y[lc], -9.205e-4)

    assert_almost_equal(node_3.results.reaction_force_x[lc], 11.131, decimal=3)
    assert_almost_equal(node_3.results.reaction_force_z[lc], -30.869, decimal=3)

    assert_almost_equal(node_4.results.reaction_force_x[lc], 0, decimal=3)
    assert_almost_equal(node_4.results.reaction_force_z[lc], 10.909, decimal=3)
    assert_almost_equal(node_4.results.reaction_moment_y[lc], 43.633, decimal=3)

    assert_array_almost_equal(
        member_12.results.min_max_axial_forces[lc], [11.131, 11.131], decimal=3
    )
    assert_array_almost_equal(
        member_12.results.min_max_shear_forces_z[lc], [19.960, 19.960], decimal=3
    )
    assert_array_almost_equal(
        member_12.results.min_max_bending_moments_y[lc], [-35.317, 44.523], decimal=3
    )

    assert_array_almost_equal(
        member_32.results.min_max_axial_forces[lc], [19.960, 19.960], decimal=3
    )
    assert_array_almost_equal(
        member_32.results.min_max_shear_forces_z[lc], [-11.131, -11.131], decimal=3
    )
    assert_array_almost_equal(
        member_32.results.min_max_bending_moments_y[lc], [-44.523, 0], decimal=3
    )

    assert_array_almost_equal(
        member_34.results.min_max_axial_forces[lc], [0, 0], decimal=3
    )
    assert_array_almost_equal(
        member_34.results.min_max_shear_forces_z[lc], [-10.909, -10.909], decimal=3
    )
    assert_array_almost_equal(
        member_34.results.min_max_bending_moments_y[lc], [-43.633, 0], decimal=3
    )


def test_example_5_3_combinations() -> None:
    """[2, Ex. 5.3] Frame under thermal load with prescribed displacements."""
    material = Material("test", 20.0e3, 0.2, 12.0e-6, 0)
    section = Section("test", 5, 5, 5, 1, 1, 1, 0.3, 0.3)

    model = FrameXZModel()

    node_1 = model.add_node(
        "1", [0, 0, 0], fixity=["fixed", "free", "fixed", "free", "fixed", "free"]
    )
    node_2 = model.add_node("2", [4, 0, 0])
    node_3 = model.add_node(
        "3", [4, 0, -4], fixity=["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_4 = model.add_node(
        "4", [8, 0, -4], fixity=["fixed", "free", "fixed", "free", "fixed", "free"]
    )

    member_12 = model.add_member("1-2", "navier", [node_1, node_2], material, section)
    member_32 = model.add_member(
        "3-2", "navier", [node_3, node_2], material, section, ["fixed", "fixed"]
    )
    member_34 = model.add_member(
        "3-4", "navier", [node_3, node_4], material, section, ["hinged", "fixed"]
    )

    lc1 = model.add_load_case("LC1")
    f1 = 2
    member_32.add_thermal_load(np.array([15, 0, 50]) / f1, lc1)

    lc2 = model.add_load_case("LC2")
    f2 = -5
    node_1.add_prescribed_displacement(0.005 / f2, "z", lc2)

    lc3 = model.add_load_case("LC3")
    f3 = 13
    node_4.add_prescribed_displacement(np.deg2rad(10 / 60) / f3, "ry", lc3)

    comb = model.add_load_case_combination("CO1", {lc1: f1, lc2: f2, lc3: f3})

    solver = LinearStaticSolver(model)
    solver.solve()

    assert_almost_equal(node_1.results.reaction_force_x[comb], -11.131, decimal=3)
    assert_almost_equal(node_1.results.reaction_force_z[comb], 19.960, decimal=3)
    assert_almost_equal(node_1.results.reaction_moment_y[comb], -35.317, decimal=3)

    assert_almost_equal(node_2.results.translation_x[comb], 4.452e-4)
    assert_almost_equal(node_2.results.translation_z[comb], 1.518e-3, decimal=6)
    assert_almost_equal(node_2.results.rotation_y[comb], -9.205e-4)

    assert_almost_equal(node_3.results.reaction_force_x[comb], 11.131, decimal=3)
    assert_almost_equal(node_3.results.reaction_force_z[comb], -30.869, decimal=3)

    assert_almost_equal(node_4.results.reaction_force_x[comb], 0, decimal=3)
    assert_almost_equal(node_4.results.reaction_force_z[comb], 10.909, decimal=3)
    assert_almost_equal(node_4.results.reaction_moment_y[comb], 43.633, decimal=3)

    assert_array_almost_equal(
        member_12.results.min_max_axial_forces[comb], [11.131, 11.131], decimal=3
    )
    assert_array_almost_equal(
        member_12.results.min_max_shear_forces_z[comb], [19.960, 19.960], decimal=3
    )
    assert_array_almost_equal(
        member_12.results.min_max_bending_moments_y[comb], [-35.317, 44.523], decimal=3
    )

    assert_array_almost_equal(
        member_32.results.min_max_axial_forces[comb], [19.960, 19.960], decimal=3
    )
    assert_array_almost_equal(
        member_32.results.min_max_shear_forces_z[comb], [-11.131, -11.131], decimal=3
    )
    assert_array_almost_equal(
        member_32.results.min_max_bending_moments_y[comb], [-44.523, 0], decimal=3
    )

    assert_array_almost_equal(
        member_34.results.min_max_axial_forces[comb], [0, 0], decimal=3
    )
    assert_array_almost_equal(
        member_34.results.min_max_shear_forces_z[comb], [-10.909, -10.909], decimal=3
    )
    assert_array_almost_equal(
        member_34.results.min_max_bending_moments_y[comb], [-43.633, 0], decimal=3
    )


def test_example_5_4() -> None:
    """[2, Ex. 5.4] Symmetric frame under force load with prescribed displacements."""
    material = Material("test", 30.0e6, 0.2, 12.0e-6, 0)
    section = Section("test", 4.0e-3, 4.0e-3, 4.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1, 1)

    model = FrameXZModel()

    fixed = ["fixed", "free", "fixed", "free", "fixed", "free"]
    vertical_roller = ["fixed", "free", "free", "free", "free", "free"]

    node_1 = model.add_node("1", [0, 0, 0], fixity=fixed)
    node_2 = model.add_node("2", [6, 0, 0], fixity=fixed)
    node_3 = model.add_node("3", [12, 0, 0], fixity=fixed)
    node_4 = model.add_node("4", [3, 0, 4], fixity=vertical_roller)
    node_5 = model.add_node("5", [6, 0, 4])
    node_6 = model.add_node("6", [9, 0, 4], fixity=vertical_roller)

    member_14 = model.add_member("1-4", "navier", [node_1, node_4], material, section)
    member_45 = model.add_member(
        "4-5", "navier", [node_4, node_5], material, section, hinges=["fixed", "hinged"]
    )
    member_25 = model.add_member(
        "2-5", "navier", [node_2, node_5], material, section, hinges=["fixed", "hinged"]
    )
    member_56 = model.add_member(
        "5-6", "navier", [node_5, node_6], material, section, hinges=["hinged", "fixed"]
    )
    member_63 = model.add_member("6-3", "navier", [node_6, node_3], material, section)

    lc = model.add_load_case("LC1")

    node_1.add_prescribed_displacement(-0.003, "x", lc)
    node_3.add_prescribed_displacement(-0.003, "x", lc)

    member_14.add_distributed_load([0, 0, 25, 0, 0, 25], lc, location="projection")
    member_45.add_distributed_load([0, 0, 25, 0, 0, 25], lc)
    member_56.add_distributed_load([0, 0, -25, 0, 0, -25], lc)
    member_63.add_distributed_load([0, 0, -25, 0, 0, -25], lc, location="projection")

    solver = LinearStaticSolver(model)
    solver.solve()

    assert_almost_equal(node_1.results.reaction_force_x[lc], -57.903, decimal=3)
    assert_almost_equal(node_1.results.reaction_force_z[lc], -116.803, decimal=2)
    assert_almost_equal(node_1.results.reaction_moment_y[lc], 19.211, decimal=2)

    assert_almost_equal(node_2.results.reaction_force_x[lc], 0, decimal=3)
    assert_almost_equal(node_2.results.reaction_force_z[lc], 0, decimal=2)
    assert_almost_equal(node_2.results.reaction_moment_y[lc], 0, decimal=2)

    assert_almost_equal(node_3.results.reaction_force_x[lc], -57.903, decimal=3)
    assert_almost_equal(node_3.results.reaction_force_z[lc], 116.803, decimal=2)
    assert_almost_equal(node_3.results.reaction_moment_y[lc], 19.211, decimal=2)

    assert_almost_equal(node_4.results.reaction_force_x[lc], 57.903, decimal=3)
    assert_almost_equal(node_4.results.translation_z[lc], 2.864e-3)
    assert_almost_equal(node_4.results.rotation_y[lc], 4.476e-4)

    assert_almost_equal(node_6.results.reaction_force_x[lc], 57.903, decimal=3)
    assert_almost_equal(node_6.results.translation_z[lc], -2.864e-3)
    assert_almost_equal(node_6.results.rotation_y[lc], 4.476e-4)

    assert_array_almost_equal(
        member_14.results.min_max_axial_forces[lc], [68.184, 128.184], decimal=2
    )
    assert_array_almost_equal(
        member_14.results.min_max_shear_forces_z[lc], [-23.759, 21.241], decimal=2
    )
    assert_array_almost_equal(
        member_14.results.min_max_bending_moments_y[lc], [-12.150, 19.211], decimal=2
    )

    assert_array_almost_equal(
        member_45.results.min_max_axial_forces[lc], [0, 0], decimal=2
    )
    assert_array_almost_equal(
        member_45.results.min_max_shear_forces_z[lc], [-41.805, 33.195], decimal=2
    )
    assert_array_almost_equal(
        member_45.results.min_max_bending_moments_y[lc], [-22.037, 12.916], decimal=2
    )

    assert_array_almost_equal(
        member_25.results.min_max_axial_forces[lc], [0, 0], decimal=2
    )
    assert_array_almost_equal(
        member_25.results.min_max_shear_forces_z[lc], [0, 0], decimal=2
    )
    assert_array_almost_equal(
        member_25.results.min_max_bending_moments_y[lc], [0, 0], decimal=2
    )

    assert_array_almost_equal(
        member_56.results.min_max_axial_forces[lc], [0, 0], decimal=2
    )
    assert_array_almost_equal(
        member_56.results.min_max_shear_forces_z[lc], [-41.805, 33.195], decimal=2
    )
    assert_array_almost_equal(
        member_56.results.min_max_bending_moments_y[lc], [-12.916, 22.037], decimal=2
    )

    assert_array_almost_equal(
        member_63.results.min_max_axial_forces[lc], [-128.184, -68.184], decimal=2
    )
    assert_array_almost_equal(
        member_63.results.min_max_shear_forces_z[lc], [-23.759, 21.241], decimal=2
    )
    assert_array_almost_equal(
        member_63.results.min_max_bending_moments_y[lc], [-19.211, 12.150], decimal=2
    )


def test_example_5_4_combinations() -> None:
    """[2, Ex. 5.4] Symmetric frame under force load with prescribed displacements."""
    material = Material("test", 30.0e6, 0.2, 12.0e-6, 0)
    section = Section("test", 4.0e-3, 4.0e-3, 4.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1, 1)

    model = FrameXZModel()

    fixed = ["fixed", "free", "fixed", "free", "fixed", "free"]
    vertical_roller = ["fixed", "free", "free", "free", "free", "free"]

    node_1 = model.add_node("1", [0, 0, 0], fixity=fixed)
    node_2 = model.add_node("2", [6, 0, 0], fixity=fixed)
    node_3 = model.add_node("3", [12, 0, 0], fixity=fixed)
    node_4 = model.add_node("4", [3, 0, 4], fixity=vertical_roller)
    node_5 = model.add_node("5", [6, 0, 4])
    node_6 = model.add_node("6", [9, 0, 4], fixity=vertical_roller)

    member_14 = model.add_member("1-4", "navier", [node_1, node_4], material, section)
    member_45 = model.add_member(
        "4-5", "navier", [node_4, node_5], material, section, hinges=["fixed", "hinged"]
    )
    member_25 = model.add_member(
        "2-5", "navier", [node_2, node_5], material, section, hinges=["fixed", "hinged"]
    )
    member_56 = model.add_member(
        "5-6", "navier", [node_5, node_6], material, section, hinges=["hinged", "fixed"]
    )
    member_63 = model.add_member("6-3", "navier", [node_6, node_3], material, section)

    lc1 = model.add_load_case("LC1")
    f1 = 5
    node_1.add_prescribed_displacement(-0.003 / f1, "x", lc1)

    lc2 = model.add_load_case("LC2")
    f2 = -3
    node_3.add_prescribed_displacement(-0.003 / f2, "x", lc2)

    lc3 = model.add_load_case("LC3")
    f3 = -25
    member_14.add_distributed_load(
        np.array([0, 0, 25, 0, 0, 25]) / f3, lc3, location="projection"
    )

    lc4 = model.add_load_case("LC4")
    f4 = 12.5
    member_45.add_distributed_load(np.array([0, 0, 25, 0, 0, 25]) / f4, lc4)

    lc5 = model.add_load_case("LC5")
    f5 = 0.2
    member_56.add_distributed_load(np.array([0, 0, -25, 0, 0, -25]) / f5, lc5)

    lc6 = model.add_load_case("LC6")
    f6 = 1.35
    member_63.add_distributed_load(
        np.array([0, 0, -25, 0, 0, -25]) / f6, lc6, location="projection"
    )

    comb = model.add_load_case_combination(
        "CO1", {lc1: f1, lc2: f2, lc3: f3, lc4: f4, lc5: f5, lc6: f6}
    )

    solver = LinearStaticSolver(model)
    solver.solve()

    assert_almost_equal(node_1.results.reaction_force_x[comb], -57.903, decimal=3)
    assert_almost_equal(node_1.results.reaction_force_z[comb], -116.803, decimal=2)
    assert_almost_equal(node_1.results.reaction_moment_y[comb], 19.211, decimal=2)

    assert_almost_equal(node_2.results.reaction_force_x[comb], 0, decimal=3)
    assert_almost_equal(node_2.results.reaction_force_z[comb], 0, decimal=2)
    assert_almost_equal(node_2.results.reaction_moment_y[comb], 0, decimal=2)

    assert_almost_equal(node_3.results.reaction_force_x[comb], -57.903, decimal=3)
    assert_almost_equal(node_3.results.reaction_force_z[comb], 116.803, decimal=2)
    assert_almost_equal(node_3.results.reaction_moment_y[comb], 19.211, decimal=2)

    assert_almost_equal(node_4.results.reaction_force_x[comb], 57.903, decimal=3)
    assert_almost_equal(node_4.results.translation_z[comb], 2.864e-3)
    assert_almost_equal(node_4.results.rotation_y[comb], 4.476e-4)

    assert_almost_equal(node_6.results.reaction_force_x[comb], 57.903, decimal=3)
    assert_almost_equal(node_6.results.translation_z[comb], -2.864e-3)
    assert_almost_equal(node_6.results.rotation_y[comb], 4.476e-4)

    assert_array_almost_equal(
        member_14.results.min_max_axial_forces[comb], [68.184, 128.184], decimal=2
    )
    assert_array_almost_equal(
        member_14.results.min_max_shear_forces_z[comb], [-23.759, 21.241], decimal=2
    )
    assert_array_almost_equal(
        member_14.results.min_max_bending_moments_y[comb], [-12.150, 19.211], decimal=2
    )

    assert_array_almost_equal(
        member_45.results.min_max_axial_forces[comb], [0, 0], decimal=2
    )
    assert_array_almost_equal(
        member_45.results.min_max_shear_forces_z[comb], [-41.805, 33.195], decimal=2
    )
    assert_array_almost_equal(
        member_45.results.min_max_bending_moments_y[comb], [-22.037, 12.916], decimal=2
    )

    assert_array_almost_equal(
        member_25.results.min_max_axial_forces[comb], [0, 0], decimal=2
    )
    assert_array_almost_equal(
        member_25.results.min_max_shear_forces_z[comb], [0, 0], decimal=2
    )
    assert_array_almost_equal(
        member_25.results.min_max_bending_moments_y[comb], [0, 0], decimal=2
    )

    assert_array_almost_equal(
        member_56.results.min_max_axial_forces[comb], [0, 0], decimal=2
    )
    assert_array_almost_equal(
        member_56.results.min_max_shear_forces_z[comb], [-41.805, 33.195], decimal=2
    )
    assert_array_almost_equal(
        member_56.results.min_max_bending_moments_y[comb], [-12.916, 22.037], decimal=2
    )

    assert_array_almost_equal(
        member_63.results.min_max_axial_forces[comb], [-128.184, -68.184], decimal=2
    )
    assert_array_almost_equal(
        member_63.results.min_max_shear_forces_z[comb], [-23.759, 21.241], decimal=2
    )
    assert_array_almost_equal(
        member_63.results.min_max_bending_moments_y[comb], [-19.211, 12.150], decimal=2
    )
