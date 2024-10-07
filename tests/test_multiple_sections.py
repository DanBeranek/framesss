import numpy as np
from numpy.ma.testutils import assert_array_almost_equal

from framesss.fea.models.frame_xz import FrameXZModel
from framesss.pre.material import Material
from framesss.pre.section import PolygonalSection
from framesss.solvers.linear_static import LinearStaticSolver


def test_multiple_sections() -> None:
    model_1 = FrameXZModel()
    node_1 = model_1.add_node(
        "N1", [0, 0, 0], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model_1.add_node(
        "N2", [5, 0, 0], ["free", "free", "free", "free", "free", "free"]
    )
    node_3 = model_1.add_node(
        "N3", [10, 0, 0], ["free", "free", "fixed", "free", "free", "free"]
    )

    section_1 = PolygonalSection(
        label="S1",
        points=[[0, 0], [0, 1], [1, 1], [1, 0]],
        material=Material("M1", 200e9, 0.3, 7850, 0),
    )
    section_2 = PolygonalSection(
        label="S1",
        points=[[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]],
        material=Material("M1", 200e9, 0.3, 7850, 0),
    )

    member_1 = model_1.add_member("M1", "navier", [node_1, node_2], section_1)
    member_2 = model_1.add_member("M2", "navier", [node_2, node_3], section_2)

    LC1 = model_1.add_load_case("LC1")
    member_1.add_distributed_load(
        [0, 0, -30000, 0, 0, -30000], LC1
    )
    member_2.add_distributed_load(
        [0, 0, -30000, 0, 0, -30000], LC1
    )

    model_2 = FrameXZModel()
    node_1 = model_2.add_node(
        "N1", [0, 0, 0], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model_2.add_node(
        "N2", [10, 0, 0], ["free", "free", "fixed", "free", "free", "free"]
    )
    member_3 = model_2.add_member("M1", "navier", [node_1, node_2], section_1)
    member_3.define_sections(
        sections={
            (0, 5): section_1,
            (5, 10): section_2,
        }
    )

    LC2 = model_2.add_load_case("LC2")
    member_3.add_distributed_load(
        [0, 0, -30000, 0, 0, -30000], LC2
    )

    model_1_solver = LinearStaticSolver(model_1)
    model_2_solver = LinearStaticSolver(model_2)

    model_1_solver.solve()
    model_2_solver.solve()

    assert_array_almost_equal(
        np.concatenate((
            member_1.results.bending_moments_y[LC1],
            member_2.results.bending_moments_y[LC1],
        )),
        member_3.results.bending_moments_y[LC2],
    )

    assert_array_almost_equal(
        np.concatenate((
            member_1.results.shear_forces_z[LC1],
            member_2.results.shear_forces_z[LC1],
        )),
        member_3.results.shear_forces_z[LC2],
    )

    assert_array_almost_equal(
        np.concatenate((
            member_1.results.axial_forces[LC1],
            member_2.results.axial_forces[LC1],
        )),
        member_3.results.axial_forces[LC2],
    )


def test_get_section() -> None:
    section_1 = PolygonalSection(
        label="S1",
        points=[[0, 0], [0, 1], [1, 1], [1, 0]],
        material=Material("M1", 200e9, 0.3, 7850, 0),
    )
    section_2 = PolygonalSection(
        label="S1",
        points=[[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]],
        material=Material("M1", 200e9, 0.3, 7850, 0),
    )

    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [0, 0, 0], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node(
        "N2", [10, 0, 0], ["free", "free", "fixed", "free", "free", "free"]
    )
    member_3 = model.add_member("M1", "navier", [node_1, node_2], section_1)

    assert member_3.get_section(x=0.0) == section_1
    assert member_3.get_section(x=5.0) == section_1
    assert member_3.get_section(x=10.0) == section_1

    member_3.define_sections(
        sections={
            (0, 5): section_1,
            (5, 10): section_2,
        }
    )

    assert member_3.get_section(x=0.0) == section_1
    assert member_3.get_section(x=2.5) == section_1
    assert member_3.get_section(x=5.0) == section_1
    assert member_3.get_section(x=7.5) == section_2
    assert member_3.get_section(x=10.0) == section_2



