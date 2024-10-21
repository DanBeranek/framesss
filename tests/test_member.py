import numpy as np
from numpy.testing import assert_array_equal

from framesss.enums import CoordinateDefinition
from framesss.fea.node import Node
from framesss.pre.section import PolygonalSection
from framesss.pre.material import Material
from framesss.fea.analysis.frame_xz_analysis import FrameXZAnalysis
from framesss.pre.member_1d import Member1D
from framesss.fea.models.frame_xz import FrameXZModel
from framesss.solvers.linear_static import LinearStaticSolver


def test_sorting_of_middle_nodes():
    N1 = Node(
        label="N1",
        coords=[0, 0, 0],
        fixity=["fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        spring_stiffness=[0, 0, 0, 0, 0, 0],
    )
    N2 = Node(
        label="N2",
        coords=[5, 0, 0],
        fixity=["fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        spring_stiffness=[0, 0, 0, 0, 0, 0],
    )

    section = PolygonalSection(
        label="S1",
        points=[[0, 0], [0, 1], [1, 1], [1, 0]],
        material=Material("M1", 200e9, 0.3, 7850, 0),
    )

    member = Member1D(
        label="M1",
        element_type="navier",
        nodes=[N1, N2],
        section=section,
        hinges=["fixed", "fixed"],
        auxiliary_vector_xy_plane=np.array([0, 0, 1]),
        analysis=FrameXZAnalysis(),
    )

    assert member.nodes == [N1, N2]

    N3 = member.add_node(
        label="N3",
        x=0.1,
        fixity=["fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        spring_stiffness=[0, 0, 0, 0, 0, 0],
    )

    assert member.nodes == [N1, N3, N2]

    N4 = member.add_node(
        label="N4",
        x=0.8,
        fixity=["fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        spring_stiffness=[0, 0, 0, 0, 0, 0],
    )

    assert member.nodes == [N1, N3, N4, N2]

    N5 = member.add_node(
        label="N5",
        x=0.6,
        fixity=["fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        spring_stiffness=[0, 0, 0, 0, 0, 0],
    )

    assert member.nodes == [N1, N3, N5, N4, N2]

    N6 = member.add_node(
        label="N6",
        x=0.05,
        fixity=["fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        spring_stiffness=[0, 0, 0, 0, 0, 0],
    )

    assert member.nodes == [N1, N6, N3, N5, N4, N2]


def test_floating_point_precision():
    """
    Because of floating point precision, the members have incorrect lengths.
        - 1st: 5.8
        - 2nd: 4.000000000000001
        - 3rd: 3.6999999999999993
        - 4th: 3.8000000000000007

    When someone assign new load to the member, the members are wrongly discretized.
    This mean we have to implement changes to the following methods:
        - define_sections()
        - add_node()
        - add_point_load()
        - add_distributed_load()
        - add_thermal_load()
        - discretize()
    """
    model = FrameXZModel()

    node_1 = model.add_node(
        label="N1",
        coords=[0, 0, 0],
        fixity=["fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
    )

    nodes = [node_1]
    lengths = [5.8, 4.0, 3.7, 3.8]

    x_curr = 0.0
    for i, length in enumerate(lengths):
        x_curr += length
        node = model.add_node(
            label=f"N{i+2}",
            coords=[x_curr, 0, 0],
            fixity=["fixed", "fixed", "fixed", "fixed", "fixed", "fixed"],
        )
        nodes.append(node)

    section = PolygonalSection(
        label="S1",
        points=[[0, 0], [0, 1], [1, 1], [1, 0]],
        material=Material("M1", 200e9, 0.3, 7850, 0),
    )

    members = []
    for i, (start_node, end_node) in enumerate(zip(nodes[:-1], nodes[1:])):
        member = model.add_member(
            label=f"M{i+1}",
            element_type="navier",
            nodes=[start_node, end_node],
            section=section,
            hinges=["fixed", "fixed"],
        )
        members.append(member)

    load_case = model.add_load_case(label="LC1")
    members[2].add_distributed_load(
        load_case=load_case,
        load_components=[0, -10e3, 0, 0, 0, 0],
        x_start=0.0,
        x_end=3.7,
        coordinate_definition=CoordinateDefinition.ABSOLUTE,
    )

    node_6 = members[3].add_node(
        label="N6",
        x=0.5,  # 0.5*3.8 = 1.9
        coordinate_definition=CoordinateDefinition.RELATIVE,
    )

    members[3].add_distributed_load(
        load_case=load_case,
        load_components=[0, -10e3, 0, 0, 0, 0],
        x_start=node_6.coords[0]-members[3].nodes[0].coords[0],
        x_end=3.8+6e-8,
        coordinate_definition=CoordinateDefinition.ABSOLUTE,
    )

    solver = LinearStaticSolver(model=model)
    solver.solve()

    assert members[2].distributed_loads[0].x_start == 0.0
    assert members[2].distributed_loads[0].x_end != 3.7
    assert members[2].distributed_loads[0].x_end == 3.6999999999999993
    assert len(members[2].generated_elements) == 1
    assert len(members[2].generated_nodes) == 2

    assert members[3].distributed_loads[0].x_start == 1.9000000000000004
    assert members[3].distributed_loads[0].x_end != 3.8
    assert members[3].distributed_loads[0].x_end == 3.8000000000000007
    assert len(members[3].generated_elements) == 2
    assert len(members[3].generated_nodes) == 3
