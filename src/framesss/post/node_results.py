from __future__ import annotations

from typing import TYPE_CHECKING

from framesss.enums import SupportFixity
from framesss.utils import DictProxy

if TYPE_CHECKING:
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

        self._reaction_force_x: dict[LoadCase, float] = {}
        self._reaction_force_y: dict[LoadCase, float] = {}
        self._reaction_force_z: dict[LoadCase, float] = {}
        self._reaction_moment_x: dict[LoadCase, float] = {}
        self._reaction_moment_y: dict[LoadCase, float] = {}
        self._reaction_moment_z: dict[LoadCase, float] = {}

        # Initialize or reset dictionaries based on node fixity.
        self.initialize_reaction_dictionaries()

    def __repr__(self) -> str:
        """Return a string representation of the NodeResults object."""
        return f"{self.__class__.__name__}(node={self.node.label})"

    def initialize_reaction_dictionaries(self) -> None:
        """
        Initialize or reset the reaction force and moment dictionaries for the node.

        This method iterates through the node's fixity attributes to determine whether
        each DoF is fixed or has a spring (i.e., capable of developing reaction forces
        or moments). For DoFs that are not fixed or do not have a spring,
        their corresponding reaction force or moment dictionary is set to None, indicating
        that no reaction forces or moments should be stored for these DoFs.
        """
        for i, fixity in enumerate(self.node.fixity):
            if fixity not in FIXITIES_WITH_REACTION:
                attr_name = (
                    f"_reaction_force_{['x', 'y', 'z'][i]}"
                    if i < 3
                    else f"_reaction_moment_{['x', 'y', 'z'][i - 3]}"
                )
                setattr(self, attr_name, None)

    @property
    def reaction_force_x(self) -> DictProxy:
        """
        Provide proxy access to the reaction force in the X-direction.

        This property returns a `DictProxy` instance that manages access to the underlying
        dictionary storing reaction forces in the x-direction for different load cases.
        It allows for dynamic interaction with the node's reaction force data, supporting
        operations like setting, getting, and deleting reaction force values.

        :return: A `DictProxy` instance for the reaction force in the X-direction.
        """
        return DictProxy(self, "_reaction_force_x")

    @property
    def reaction_force_y(self) -> DictProxy:
        """
        Provide proxy access to the reaction force in the Y-direction.

        :return: A `DictProxy` instance for the reaction force in the Y-direction.
        """
        return DictProxy(self, "_reaction_force_y")

    @property
    def reaction_force_z(self) -> DictProxy:
        """
        Provide proxy access to the reaction force in the Z-direction.

        :return: A `DictProxy` instance for the reaction force in the Z-direction.
        """
        return DictProxy(self, "_reaction_force_z")

    @property
    def reaction_moment_x(self) -> DictProxy:
        """
        Provide proxy access to the reaction moment about X-axis.

        :return: A `DictProxy` instance for the reaction moment about X-axis.
        """
        return DictProxy(self, "_reaction_moment_x")

    @property
    def reaction_moment_y(self) -> DictProxy:
        """
        Provide proxy access to the reaction moment about Y-axis.

        :return: A `DictProxy` instance for the reaction moment about Y-axis.
        """
        return DictProxy(self, "_reaction_moment_y")

    @property
    def reaction_moment_z(self) -> DictProxy:
        """
        Provide proxy access to the reaction moment about Z-axis.

        :return: A `DictProxy` instance for the reaction moment about Z-axis.
        """
        return DictProxy(self, "_reaction_moment_z")
