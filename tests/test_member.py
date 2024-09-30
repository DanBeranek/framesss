import numpy as np
from numpy.testing import assert_array_equal

from framesss.fea.node import Node
from framesss.pre.section import PolygonalSection
from framesss.pre.material import Material
from framesss.fea.analysis.frame_xz_analysis import FrameXZAnalysis
from framesss.pre.member_1d import Member1D


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
