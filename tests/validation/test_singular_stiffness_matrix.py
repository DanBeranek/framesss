import pytest

from framesss.enums import BeamConnection
from framesss.errors import SingularMatrixError
from framesss.fea.models.frame_xz import FrameXZModel
from framesss.pre.material import Material
from framesss.pre.section import Section
from framesss.solvers.linear_static import LinearStaticSolver


DUMMY_MAT = Material("foo", 1, 1, 1, 1)
DUMMY_SEC = Section("bar", 1, 1, 1, 1, 1, 1, 1, 1, DUMMY_MAT)


def test_singular_stiffness_matrix() -> None:
    model = FrameXZModel()
    node_1 = model.add_node(
        "N1", [0.0, 0.0, 0.0], ["fixed", "free", "fixed", "free", "free", "free"]
    )
    node_2 = model.add_node("N2", [5.0, 0.0, 0.0])
    node_3 = model.add_node(
        "N3", [10.0, 0.0, 0.0], ["fixed", "free", "fixed", "free", "free", "free"]
    )

    member_1 = model.add_member(
        "M1",
        "navier",
        [node_1, node_2],
        DUMMY_SEC,
        hinges=[BeamConnection.CONTINUOUS_END, BeamConnection.HINGED_END],
    )
    member_2 = model.add_member("M2", "navier", [node_2, node_3], DUMMY_SEC)

    load_case = model.add_load_case("LC1")
    solver = LinearStaticSolver(model)

    # Load in [1] is applied in the negative z-direction
    member_2.add_distributed_load([0, 0, -5, 0, 0, -5], load_case)

    with pytest.raises(SingularMatrixError) as exc_info:
        solver.solve()
