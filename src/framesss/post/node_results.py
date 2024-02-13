from __future__ import annotations

from typing import TYPE_CHECKING

from framesss.enums import SupportFixity

if TYPE_CHECKING:
    import numpy as np

    from framesss.fea.node import Node
    from framesss.pre.cases import LoadCase

FIXITIES_WITH_REACTION = [SupportFixity.FIXED_DOF, SupportFixity.SPRING_DOF]


class NodeResults:
    """Class for storing results from finite element analysis."""

    def __init__(self, node: Node) -> None:
        """Init the NodeResults class."""
        self.node = node

        self.translation_x: dict[LoadCase, float] = {}
        self.translation_y: dict[LoadCase, float] = {}
        self.translation_z: dict[LoadCase, float] = {}

        self.rotation_x: dict[LoadCase, float] = {}
        self.rotation_y: dict[LoadCase, float] = {}
        self.rotation_z: dict[LoadCase, float] = {}

        self.reaction_force_x: dict[LoadCase, np.float64] | None = (
            {} if node.fixity[0] in FIXITIES_WITH_REACTION else None
        )
        self.reaction_force_y: dict[LoadCase, float] | None = (
            {} if node.fixity[1] in FIXITIES_WITH_REACTION else None
        )
        self.reaction_force_z: dict[LoadCase, float] | None = (
            {} if node.fixity[2] in FIXITIES_WITH_REACTION else None
        )
        self.reaction_moment_x: dict[LoadCase, float] | None = (
            {} if node.fixity[3] in FIXITIES_WITH_REACTION else None
        )
        self.reaction_moment_y: dict[LoadCase, float] | None = (
            {} if node.fixity[4] in FIXITIES_WITH_REACTION else None
        )
        self.reaction_moment_z: dict[LoadCase, float] | None = (
            {} if node.fixity[5] in FIXITIES_WITH_REACTION else None
        )

    def __repr__(self) -> str:
        """Return a string representation of the NodeResults object."""
        return f"{self.__class__.__name__}(node={self.node.label})"
