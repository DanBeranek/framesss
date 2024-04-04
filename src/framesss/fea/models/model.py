from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from framesss.enums import BeamConnection
from framesss.enums import Element1DType
from framesss.enums import SupportFixity
from framesss.fea.node import Node
from framesss.pre.cases import EnvelopeCombination
from framesss.pre.cases import LoadCase
from framesss.pre.cases import LoadCaseCombination
from framesss.pre.member_1d import Member1D

if TYPE_CHECKING:
    import numpy.typing as npt

    from framesss.fea.analysis.analysis import Analysis
    from framesss.fea.element_1d import Element1D
    from framesss.pre.material import Material
    from framesss.pre.section import Section


class Model:
    """
    Class represent the entire structural analysis model.

    This class encapsulates the analysis context, materials, sections, nodes, members, load cases,
    and other essential elements for structural analysis within an analysis.

    :param analysis: A reference to an instance of the :class:`Analysis` class, which specifies
                           the analytical approach and context for the model (e.g., type of analysis,
                           dimensional considerations).
    :ivar materials: A set of :class:`Material` instances, representing the materials used in the analysis.
    :ivar sections: A set of :class:`Section` instances, detailing the cross-sectional properties of members.
    :ivar nodes: A set of :class:`Node` instances, defining the points of interest within the model
                 where members connect or loads are applied.
    :ivar members: A set of :class:`Member1D` instances, representing the one-dimensional structural
                   members (e.g., beams, columns) in the model.
    :ivar load_cases: A set of :class:`LoadCase` instances, each defining a unique set
                      of loading conditions to be analyzed.
    :ivar load_combinations: A set of :class:`LoadCaseCombination` instances, defining
                             load_cases of load cases for analysis.
    :ivar elements: A set of :class:`Element1D` instances.
    :ivar neq_free: Number of equations corresponding to free DoFs.
    :ivar neq_fixed: Number of equations corresponding to fixed DoFs.
    :ivar neq_spring: Number of equations corresponding to spring DoFs.
    :ivar k_global: Global stiffness matrix for the entire model.
    :ivar spring_stiffness_global: List containing global spring stiffness coefficients.
    :ivar dof_connectivity_matrix: Matrix defining the global numbering of DoFs for each node.

    The ``Model`` class plays a pivotal role in setting up and managing the data necessary for performing
    structural analysis, providing methods to add, remove, or query components of the model.
    """

    def __init__(self, analysis: Analysis) -> None:
        """Init the Model object."""
        self.analysis = analysis
        self.materials: set[Material] = set()
        self.sections: set[Section] = set()
        self.nodes: set[Node] = set()
        self.members: set[Member1D] = set()
        self.load_cases: set[LoadCase] = set()
        self.load_combinations: set[LoadCaseCombination] = set()
        self.envelopes: set[EnvelopeCombination] = set()
        self.elements: set[Element1D] = set()

        self.neq_free: int = 0
        self.neq_fixed: int = 0
        self.neq_spring: int = 0

        self.k_global: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.spring_stiffness_global: npt.NDArray[np.float64] = np.empty(
            0, dtype=np.float64
        )
        self.dof_connectivity_matrix: npt.NDArray[np.int64] = np.empty(
            0, dtype=np.int64
        )

    def __repr__(self) -> str:
        """Return string representation of Model object."""
        return (
            f"{self.__class__.__name__}("
            f"analysis={self.analysis.__class__.__name__})"
        )

    @property
    def neq(self) -> int:
        """Returns number of equations."""
        return self.number_of_nodes * self.analysis.n_dof_per_node

    @property
    def number_of_nodes(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)

    @property
    def number_of_elements(self) -> int:
        """Return number of elements."""
        return len(self.elements)

    @staticmethod
    def add_node(
        label: str,
        coords: list[float],
        fixity: list[str] | tuple[str, ...] = ("free",) * 6,
        spring_stiffness: list[float] | tuple[float, ...] = (0.0,) * 6,
    ) -> Node:
        """
        Create and return new :class:`Node` instance.

        :param label: Unique user-defined label for the node.
        :param coords: List of coordinates in the global system [X, Y, Z].
        :param fixity: A list specifying the essential boundary conditions for each degree of freedom (DoF)
                       at the node. Conditions can be 'free', 'fixed', or 'spring'.
        :param spring_stiffness: A list of spring stiffness coefficients applicable when the fixity is set
                                 to 'spring' for corresponding DoFs. The list should follow the order
                                 [kx, ky, kz, krx, kry, krz], representing translational and rotational spring
                                 stiffness values in the X, Y, and Z directions, respectively.
        """
        fixities = [SupportFixity(fix) for fix in fixity]
        spring_stiff = [stiff for stiff in spring_stiffness]

        coords = [coords[i] if i < len(coords) else 0 for i in range(3)]
        new_node = Node(label, coords, fixities, spring_stiff)
        return new_node

    def add_member(
        self,
        label: str,
        element_type: str,
        nodes: list[Node],
        section: Section,
        hinges: list[str] | tuple[str, str] = (BeamConnection.CONTINUOUS_END,) * 2,
    ) -> Member1D:
        """
        Create and return new :class:`Member1D` instance.

        :param label: A user-defined label for the member.
        :param element_type: Specifies the type of the element ('navier', 'timoshenko').
        :param nodes: A list of nodes at the start and the end of the member.
        :param section: The cross-section of the member.
        :param hinges: Defines the type of connections at the start and the end of the  member
                       (e.g., fixed, hinged, or semirigid) to model the rotational stiffness accurately.
        """
        elem_type = Element1DType(element_type)
        hngs = [BeamConnection(hng) for hng in hinges]

        aux = self.analysis.get_auxiliary_vector_in_local_xy_plane(nodes)
        new_member = Member1D(
            label,
            elem_type,
            nodes,
            section,
            hngs,
            aux,
            self.analysis,
        )
        self.members.add(new_member)
        return new_member

    def add_load_case(self, label: str) -> LoadCase:
        """
        Add and return new :class:`LoadCase` instance.

        :param label: Unique user-defined label of the load case.
        """
        new_case = LoadCase(label)
        self.load_cases.add(new_case)
        return new_case

    def add_load_case_combination(
        self, label: str, combination: dict[LoadCase, float]
    ) -> LoadCaseCombination:
        """
        Add and return new :class:`LoadCaseCombination` instance.

        :param label: Unique user-defined label of the load combination.
        :param combination: Dictionary mapping :class:`LoadCase` instances
                            to their scaling factors.
        """
        new_combination = LoadCaseCombination(label, combination)
        self.load_combinations.add(new_combination)
        return new_combination

    def add_envelope(
        self, label: str, cases: list[LoadCase | LoadCaseCombination]
    ) -> EnvelopeCombination:
        """
        Add and return new :class:`EnvelopeCombination` instance.

        :param label: Unique user-defined label of the load combination.
        :param cases:  A list of :class:`LoadCase` and :class:`LoadCombination`.
        """
        new_envelope = EnvelopeCombination(label, cases)
        self.envelopes.add(new_envelope)
        return new_envelope

    def discretize_members(self) -> None:
        """
        Discretize all members in model and assign IDs to nodes and elements.

        This method iterates over each member in the model, calling its `discretize`
        method to divide it into finite elements. After discretization, it assigns
        a unique ID to each node and element in the model.
        """
        for member in self.members:
            member.discretize(self)

        for i, node in enumerate(self.nodes):
            node.id = i

        for i, element in enumerate(self.elements):
            element.id = i

    def setup_fictitious_rotation_constraints(self, fict: bool) -> None:
        """
        Adjust rotation constraints on nodes to ensure global stability.

        For nodes exclusively connected by hinged elements, this method either adds or removes
        fictitious rotation constraints. These constraints are used to prevent a singular global
        stiffness matrix by artificially restricting rotational degrees of freedom, enhancing
        stability for the finite element analysis.

        :param fict: A flag indicating whether to add (True) or remove (False) the fictitious rotation constraints.
        """
        for node in self.nodes:
            # Create fictitious rotation constraints (before fea process)
            if fict:
                tot, hng = node.get_elements_incidence(self)

                if tot == hng:
                    # Change to fictfixed constrained.
                    if node.fixity[3] == SupportFixity(SupportFixity.FREE_DOF):
                        node.fixity[3] = SupportFixity(SupportFixity.FICTFIXED_DOF)

                    if node.fixity[4] == SupportFixity(SupportFixity.FREE_DOF):
                        node.fixity[4] = SupportFixity(SupportFixity.FICTFIXED_DOF)

                    if node.fixity[5] == SupportFixity(SupportFixity.FREE_DOF):
                        node.fixity[5] = SupportFixity(SupportFixity.FICTFIXED_DOF)

            # Remove fictitious rotation constraints (after fea process)
            elif not fict:
                if node.fixity[3] == SupportFixity(SupportFixity.FICTFIXED_DOF):
                    node.fixity[3] = SupportFixity(SupportFixity.FREE_DOF)

                if node.fixity[4] == SupportFixity(SupportFixity.FICTFIXED_DOF):
                    node.fixity[4] = SupportFixity(SupportFixity.FREE_DOF)

                if node.fixity[5] == SupportFixity(SupportFixity.FICTFIXED_DOF):
                    node.fixity[5] = SupportFixity(SupportFixity.FREE_DOF)

    def assemble_dof_indices(self) -> None:
        """
        Assign equation numbers to degrees of freedom (DoFs) based on their type.

        This method iterates through each degree of freedom for every node in the model to assign
        equation numbers. Free DoFs are numbered first, followed by spring and fixed DoFs.
        The assignment is reflected in the `dof_connectivity_matrix`, where each DoF is tagged
        with its corresponding equation number, facilitating the assembly of global matrices for FEA.

        Free DoFs are assigned consecutive numbers starting from 0. Spring DoFs continue numbering
        from the last free DoF, and fixed DoFs follow, starting after the last spring DoF. This
        systematic numbering is essential for partitioning the global stiffness matrix during the analysis process.
        """
        # Initialize equation numbers for free, fixed and spring DoFs
        count_free = 0
        count_spring = self.neq_free
        count_fixed = self.neq_free + self.neq_spring

        # Check if each DoF is free, fixed or spring to increment equation
        # number and store it in the dof_connectivity_matrix matrix
        for n in range(self.number_of_nodes):
            for k in range(self.analysis.n_dof_per_node):
                # TODO: StrEnum didn't work here for EssentialBoundaryCondition
                if self.dof_connectivity_matrix[k, n] == 0:
                    self.dof_connectivity_matrix[k, n] = count_free
                    count_free += 1
                elif self.dof_connectivity_matrix[k, n] == 2:
                    self.dof_connectivity_matrix[k, n] = count_spring
                    count_spring += 1
                else:
                    self.dof_connectivity_matrix[k, n] = count_fixed
                    count_fixed += 1

    def assemble_global_dofs(self) -> None:
        """Assembles gather vectors of elements (global_dofs) that store member d.o.f.'s. equation numbers."""
        for elem in self.elements:
            self.analysis.setup_dof_mapping(self, elem)
