from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from framesss.enums import BeamConnection
from framesss.enums import DoF
from framesss.enums import SupportFixity
from framesss.fea.boundary_conditions.nodal_load import NodalLoad
from framesss.fea.boundary_conditions.prescribed_displacement import (
    PrescribedDisplacement,
)
from framesss.post.node_results import NodeResults

if TYPE_CHECKING:
    import numpy.typing as npt

    from framesss.fea.models.model import Model
    from framesss.pre.cases import LoadCase


SUPPORT_FIXITIES = [
    SupportFixity.FREE_DOF,
    SupportFixity.FIXED_DOF,
    SupportFixity.SPRING_DOF,
    SupportFixity.FICTFIXED_DOF,
]


class Node:
    """
    Represent a node within a finite element analysis (FEA) model.

    A node is a joint between two or more elements, or any member end, used to discretize the model.
    It is always considered as a 3D entity which may have applied loads or prescribed displacements.

    :param label: A user-defined label for the node.
    :param coords: List of coordinates in the global system [X, Y, Z].
    :param fixity: A list specifying the essential boundary conditions for each degree of freedom (DoF)
                   at the node. Conditions can be 'free', 'fixed', or 'spring'.
    :param spring_stiffness: A list of spring stiffness coefficients applicable when the fixity is set
                             to 'spring' for corresponding DoFs. The list should follow the order
                             [kx, ky, kz, krx, kry, krz], representing translational and rotational spring
                             stiffness values in the X, Y, and Z directions, respectively.

    :ivar id: An automatically assigned identification number for the node, unique within a given FEA model.
    :ivar loads: A dictionary mapping LoadCase instances to NodalLoad objects, representing loads applied to the node.
    :ivar results: An instance of NodeResults, storing analysis results related to the node.
    :ivar global_dofs: A list of global degrees of freedom indices associated with this node, set during model assembly.
    """

    def __init__(
        self,
        label: str,
        coords: list[float] | npt.NDArray[np.float64],
        fixity: list[SupportFixity],
        spring_stiffness: list[float],
    ) -> None:
        """Init the Node class."""
        self.label = label
        self.id: None | int = None
        self.coords = coords
        self.fixity = self.validate_fixity(fixity)
        self.spring_stiffness = spring_stiffness
        self.loads: dict[LoadCase, NodalLoad] = {}
        self.results = NodeResults(self)
        self.global_dofs: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)

    def __repr__(self) -> str:
        """Return a string representation of the Node object."""
        return (
            f"{self.__class__.__name__}("
            f"label={self.label!r}, "
            f"coords={self.coords!r}, "
            f"fixity={self.fixity!r}, "
            f"spring_stiffness={self.spring_stiffness!r})"
        )

    def validate_fixity(
        self, fixity: list[SupportFixity] | list[str]
    ) -> list[SupportFixity]:
        """
        Validate the fixity conditions provided for a node.

        Checks if the fixity list has 6 elements corresponding to the degrees of freedom in 3D space
        (translations and rotations along/about X, Y, Z axes). Also verifies that each fixity condition
        is valid according to predefined SupportFixity values.

        :param fixity: A list of fixity conditions to be validated.
        :return: Same fixity list if it passes all validations.
        :raises: ValueError if the fixity list is not valid.
        """
        if len(fixity) != 6:
            raise ValueError(
                f"Fixity at node '{self.label}' must have 6 elements. Got {len(fixity)} elements."
            )

        fixities = []

        for i, fix in enumerate(fixity):
            if fix not in SUPPORT_FIXITIES:
                raise ValueError(
                    f"Invalid fixity: '{fix}' for the node: '{self.label}' at the index {i}\n"
                    f"Valid fixities are: {SUPPORT_FIXITIES}."
                )
            fixities.append(SupportFixity(fix))

        return fixities

    def get_elements_incidence(self, model: Model) -> tuple[int, int]:
        """
        Count total number of elements and number of hinged elements connected to a node.

        :param model: A reference to the instance of the :class:`Model` class containing
                      the node and its connected elements. This is used to access the
                      list of elements for incidence counting.
        :return: A tuple containing total number of elements connected to this node and
                 number of those elements that are connected via a hinged end at this node.
        """
        tot = 0
        hng = 0

        for element in model.elements:
            # check if initial node of current member is the target one
            if element.nodes[0].id == self.id:
                tot += 1
                if element.hinge_start == BeamConnection.HINGED_END:
                    hng += 1

            # check if final node of current member is the target one
            elif element.nodes[-1].id == self.id:
                tot += 1
                if element.hinge_end == BeamConnection.HINGED_END:
                    hng += 1

        return tot, hng

    # TODO: Move this to the Model class?
    def add_nodal_load(
        self,
        load_components: list[float] | npt.NDArray[np.float64],
        load_case: LoadCase,
    ) -> None:
        """
        Add point forces and moments to the node for a specified load case.

        This method updates the nodal loads for this node within the specified load case.
        The load components provided should include forces and moments in the global
        coordinate system, formatted as [Fx, Fy, Fz, Mx, My, Mz], where 'F' denotes forces
        along the respective axes and 'M' denotes moments about the respective axes.

        If the node already has an associated nodal load in the load case, this method
        updates the existing load components. If not, a new NodalLoad instance is created
        and associated with this node under the given load case.

        :param load_components: The components of the point force and moments in the format
                                [Fx, Fy, Fz, Mx, My, Mz], representing the force in the x,
                                y, and z directions and moments about x, y and z-axis.
        :param load_case: The :class:`LoadCase` to which this point force belongs.
        """
        # Check if this node already has an associated NodalLoad in the given LoadCase
        if not load_case.nodal_loads.get(self):
            # If not, create a new NodalLoad instance for this node
            load_case.nodal_loads[self] = NodalLoad()

        # Update the NodalLoad components for this node in the given LoadCase
        load_case.nodal_loads[self].load_components += load_components

    def add_prescribed_displacement(
        self,
        prescribed_displacement: float,
        degree_of_freedom: str | DoF,
        load_case: LoadCase,
    ) -> None:
        """
        Add a prescribed displacement to a specific DoF for this node under a given load case.

        Validates the degree of freedom (DoF) to ensure it corresponds to a fixed DoF and updates the
        prescribed displacement value accordingly. If no prescribed displacement exists for this node
        under the specified load case, a new instance is created.

        :param prescribed_displacement: The displacement value.
        :param degree_of_freedom: The DoF (translational/rotational).
        :param load_case: The :class:`LoadCase` to which the prescribed displacement applies.

        :raises TypeError: If `degree_of_freedom` is not a string or DoF instance.
        :raises ValueError: If attempting to add prescribed displacement to a non-fixed DoF.
        """
        if isinstance(degree_of_freedom, DoF):
            idx = degree_of_freedom.index
        elif isinstance(degree_of_freedom, str):
            idx = DoF.get_index(degree_of_freedom)
        else:
            raise TypeError(
                f"'degree_of_freedom' must be a 'str' or 'DoF̈́' instance, not {type(degree_of_freedom)}"
            )

        # Check, if the direction in which the prescribed displacement is, is fixed
        if self.fixity[idx] != SupportFixity.FIXED_DOF:
            raise ValueError(
                "Prescribed displacement can be applied only to fixed DoF."
            )

        # Check if this node already has an associated NodalLoad in the given LoadCase
        if not load_case.prescribed_displacements.get(self):
            # If not, create a new PrescribedDisplacement instance for this node
            load_case.prescribed_displacements[self] = PrescribedDisplacement()

        # Update the PrescribedDisplacement components for this node in the given LoadCase
        load_case.prescribed_displacements[self].prescribed_displacements[
            idx
        ] += prescribed_displacement
