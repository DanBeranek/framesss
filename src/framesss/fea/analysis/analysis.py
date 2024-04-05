from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from framesss.enums import SupportFixity

if TYPE_CHECKING:
    import numpy.typing as npt

    from framesss.enums import AnalysisModelType
    from framesss.enums import DoF
    from framesss.fea.boundary_conditions.element_load import ElementLoad
    from framesss.fea.element_1d import Element1D
    from framesss.fea.models.model import Model
    from framesss.fea.node import Node
    from framesss.pre.cases import EnvelopeCombination
    from framesss.pre.cases import LoadCase
    from framesss.pre.cases import LoadCaseCombination
    from framesss.pre.member_1d import Member1D


class Analysis(ABC):
    """
    Abstract base class for defining a finite element analysis (FEA).

    The :class:`Analysis` class serves as a foundational blueprint for creating specific
    types of FEA models. It is designed to project the generic 3D behavior of objects
    from the :class:`Member1D` and :class:`Node` classes into specific model behaviors
    suitable for various types of analysis.

    This class defines a set of abstract methods that must be implemented in its subclasses.
    These implementations should provide the specific properties and behaviors unique to each
    type of FEA model. The abstract nature of this class means it cannot be instantiated
    directly; instead, it should be subclassed to create specific model types such as:

      - :class:`FrameXZAnalysis`
      - :class:`FrameXYZAnalysis`
      - :class:`GridXYAnalysis`
      - :class:`TrussXYAnalysis`
      - :class:`TrussXYZAnalysis`

    Each subclass represents a different structural analysis model, encapsulating the unique
    characteristics and behaviors of that model type.

    :param analysis_type: The type of FEA model to be used, which dictates the behavior
                          of the analysis model.
    :param active_dofs: A list of active degrees of freedom for the model. These represent
                        the translational and rotational DoFs that are considered in the analysis.
    :param active_rotational_dofs: A subset of ``active_dofs`` specifying which DoFs are rotational.
    :ivar active_displacement_dofs: A subset of ``active_dofs`` specifying which DoFs are translational.
    :ivar n_dof_per_node: The number of active degrees of freedom per node.
    :ivar n_rot_dof_per_node: The number of active rotational degrees of freedom per node.
    :ivar dof_elem_axial, dof_elem_flexural_xy, dof_elem_flexural_xz, dof_elem_torsion: Lists
      that will be populated with indices corresponding to different types of DoFs in specific
      type of analysis. These are left empty at initialization and should be defined in subclasses.
    """

    def __init__(
        self,
        analysis_type: AnalysisModelType,
        active_dofs: list[DoF],
        active_rotational_dofs: list[DoF],
    ) -> None:
        """Init the Analysis class."""
        self.analysis_type = analysis_type
        self.active_dofs = [dof for dof in active_dofs]
        self.active_rotational_dofs = [dof for dof in active_rotational_dofs]
        self.active_displacement_dofs = [
            dof for dof in active_dofs if dof not in active_rotational_dofs
        ]
        self.n_dof_per_node = len(active_dofs)
        self.n_rot_dof_per_node = len(active_rotational_dofs)
        self.dof_elem_axial: list[int] = []
        self.dof_elem_flexural_xy: list[int] = []
        self.dof_elem_flexural_xz: list[int] = []
        self.dof_elem_torsion: list[int] = []

    def __repr__(self) -> str:
        """Return a string representation of the Analysis object."""
        return (
            f"{self.__class__.__name__}("
            f"analysis_type={self.analysis_type!r}, "
            f"active_dofs={self.active_dofs!r}, "
            f"active_rotational_dofs={self.active_rotational_dofs!r}, "
            f"active_displacement_dofs={self.active_displacement_dofs!r}, "
            f"n_dof_per_node={self.n_dof_per_node!r}, "
            f"n_rot_dof_per_node={self.n_rot_dof_per_node!r})"
        )

    @abstractmethod
    def get_auxiliary_vector_in_local_xy_plane(
        self, nodes: list[Node]
    ) -> npt.NDArray[np.float64]:
        """
        Calculate and return the auxiliary vector in the local xy-plane for a given list of nodes.

        The auxiliary vector lies in the local xy-plane of a member, and the cross-product
        of the auxiliary vector with the x-axis defines the local z-axis vector.

        :param nodes: A list of :class:`Node` objects representing the nodes of the
                      member for which the auxiliary vector is calculated.
        :return: The auxiliary vector in the local xy-plane. This vector is essential for
                 defining the orientation of the member in its local coordinate system.
        """
        pass

    @abstractmethod
    def get_transformation_matrix(self, member: Member1D) -> npt.NDArray[np.float64]:
        """
        Compute and return the transformation matrix for a given structural member.

        :param member: A reference to an instance of the :class:`Member1D` class.
        :return: The transformation matrix for the given member.
        """
        pass

    @abstractmethod
    def get_element_local_stiffness_matrix(
        self, element: Element1D
    ) -> npt.NDArray[np.float64]:
        """
        Assembles and returns the local stiffness matrix for a specified element.

        :param element: A reference to an instance of the :class:`Element1D` class.
        :return: The local stiffness matrix of the specified element.
        """
        pass

    @abstractmethod
    def assemble_nodal_loads(self, model: Model, load_case: LoadCase) -> None:
        """
        Add nodal load components to the global force vector for a given load case.

        This method iterates over all nodal loads defined in a given load case and adds
        their components to the global force vector.

        :param model: A reference to an instance of the :class:`Model` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        pass

    @abstractmethod
    def get_fixed_end_forces(self, load: ElementLoad) -> npt.NDArray[np.float64]:
        """
        Return the fixed end force vector for given load.

        :param load: A reference to an instance of the :class:`ElementLoad` class,
                     representing the distributed load applied to the member.
        :return: The fixed end force vector in local system for the element under the specified load.
        """
        pass

    @abstractmethod
    def assemble_internal_forces(
        self, element: Element1D, load_case: LoadCase, fel: npt.NDArray[np.float64]
    ) -> None:
        """
        Assembles contribution of an internal force vector for a given element and load case.

        :param element: A reference to an instance of the :class:`Element1D` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :param fel: The internal force vector in local system for the element.
        """
        pass

    @abstractmethod
    def get_displacement_shape_function_matrix(
        self, element: Element1D
    ) -> npt.NDArray[np.float64]:
        """
        Return displacement shape function matrix evaluated at sampling points of the element.

        :param element: A reference to an instance of the :class:`Element1D` class.
        :return: Displacement shape function matrix.
        """
        pass

    @abstractmethod
    def get_internal_displacements_from_global_analysis(
        self, n: npt.NDArray[np.float64], u_local: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Return interpolated displacements evaluated at sampling points of the element.

        This method interpolates nodal displacements to the sampling points using shape functions.

        :param n: Displacement shape function matrix.
        :param u_local: Nodal displacements obtained from global analysis in local coordinate system.
        :return: Interpolated displacements at the element sampling points.
        """
        pass

    @abstractmethod
    def get_internal_displacements_from_local_analysis(
        self, element: Element1D, load_case: LoadCase
    ) -> npt.NDArray[np.float64]:
        """
        Return displacements from loads evaluated at sampling points of the element.

        :param element: A reference to an instance of the :class:`Element1D` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :return: Displacements from loads at the element sampling points.
        """
        pass

    @abstractmethod
    def save_internal_stresses(
        self, member: Member1D, case: LoadCase | LoadCaseCombination
    ) -> None:
        """
        Compute and save the internal stresses.

        Internal stresses are: axial forces, shear forces, and bending moments,
        for a member under a specified load case. This includes both detailed distributions
        along the member and extreme values for each stress component.

        This method aggregates internal stress data from each :class:`Element1D` of the :class:`Member1D`.
        :param member: A reference to an instance of the :class:`Member1D` class.
        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        """
        pass

    @abstractmethod
    def save_envelope_stresses(
        self, member: Member1D, envelope: EnvelopeCombination
    ) -> None:
        """
        Compute and save the envelope of internal stresses.

        :param member: A reference to an instance of the :class:`Member1D` class.
        :param envelope: A reference to an instance of the :class:`EnvelopeCombination`.
        """
        pass

    @abstractmethod
    def save_internal_displacements_on_member(
        self, member: Member1D, load_case: LoadCase
    ) -> None:
        """
        Compute and save the internal displacements for a member under a specified load case.

        This method aggregates displacement data from each :class:`Element1D` of the :class:`Member1D`,

        :param member: A reference to an instance of the :class:`Member1D` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        pass

    @abstractmethod
    def save_internal_displacements_on_member_combination(
        self, member: Member1D, load_combination: LoadCaseCombination
    ) -> None:
        """
        Compute and save the internal displacements for a member under a specified load case.

        This method aggregates displacement data from each :class:`Element1D` of the :class:`Member1D`,

        :param member: A reference to an instance of the :class:`Member1D` class.
        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        """
        pass

    @abstractmethod
    def save_reactions(self, node: Node, load_case: LoadCase) -> None:
        """
        Save the reaction forces and moments for a specified node under a given load case.

        This method extracts reaction forces and moments from the global force vector for the specified
        :class:`LoadCase` and assigns them to the corresponding node results.

        :param node: A reference to an instance of the :class:`Node` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.

        Note that the method operates directly on the `node.results` attribute, updating it with
        the calculated reactions for the specified load case. If `node.fixity` for a particular
        direction is not `SupportFixity.FIXED_DOF` (i.e. reaction storage for a particular direction
        is None), this method will not attempt to save reactions in that direction.
        """
        pass

    # TODO: docstring
    @abstractmethod
    def save_reactions_combination(
        self, node: Node, load_combination: LoadCaseCombination
    ) -> None:
        """
        Save the reaction forces and moments for a specified node under a given load combination.

        This method extracts reaction forces and moments from the global force vector for the specified
        :class:`LoadCase` and assigns them to the corresponding node results.

        :param node: A reference to an instance of the :class:`Node` class.
        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.

        Note that the method operates directly on the `node.results` attribute, updating it with
        the calculated reactions for the specified load case. If `node.fixity` for a particular
        direction is not `SupportFixity.FIXED_DOF` (i.e. reaction storage for a particular direction
        is None), this method will not attempt to save reactions in that direction.
        """
        pass

    @abstractmethod
    def save_displacements(self, node: Node, load_case: LoadCase) -> None:
        """
        Save the displacements for a specified node under a given load case.

        This method extracts displacements from the global displacement vector for the
        specified :class:`LoadCase` and assigns them to the corresponding node results.

        :param node: A reference to an instance of the :class:`Node` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        pass

    # TODO: docstring
    @abstractmethod
    def save_displacements_combination(
        self, node: Node, load_combination: LoadCaseCombination
    ) -> None:
        """
        Save the displacements for a specified node under a given load combination.

        This method extracts displacements from the global displacement vector for the
        specified :class:`LoadCase` and assigns them to the corresponding node results.

        :param node: A reference to an instance of the :class:`Node` class.
        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        """

    def setup_dof_numbers(self, model: Model) -> None:
        """
        Initialize the global DoF numbering matrix and counts the total number of equations.

        The `dof_connectivity_matrix` matrix is used to track the status of each DoF for every node in the model:
            - a value of 0 indicates a free DoF,
            - a value of 1 indicates a DoF constrained by fixed support,
            - a value of 2 indicates a DoF constrained by spring support,
            - a value of -1 indicates a DoF constrained by fictitious support
              (that is not physically constrained but treated as fixed for analysis purposes).

        :param model: A reference to an instance of the :class:`Model` class.
        """
        model.dof_connectivity_matrix = np.zeros(
            [self.n_dof_per_node, model.number_of_nodes], dtype=int
        )

        model.neq_fixed = 0
        model.neq_spring = 0

        for n, node in enumerate(model.nodes):
            i = 0
            for dof in self.active_dofs:
                idx = dof.index
                # Check for fixed d.o.f.
                if (
                    node.fixity[idx] == SupportFixity.FIXED_DOF
                    or node.fixity[idx] == SupportFixity.FICTFIXED_DOF
                ):
                    model.neq_fixed += 1
                    model.dof_connectivity_matrix[i, n] = 1
                # Check for d.o.f. associated with spring
                elif node.fixity[idx] == SupportFixity.SPRING_DOF:
                    model.neq_spring += 1
                    model.dof_connectivity_matrix[i, n] = 2

                i += 1

        model.neq_free = +model.neq - model.neq_fixed - model.neq_spring

    def setup_dof_mapping(self, model: Model, element: Element1D) -> None:
        """
        Assembles global DoF indices (equation numbers) for given element.

        :param model: A reference to an instance of the :class:`Model` class.
        :param element: A reference to an instance of the :class:`Element1D` class.
        """
        element.global_dofs = np.zeros(
            [element.number_of_nodes * self.n_dof_per_node], dtype=int
        )

        for n, node in enumerate(element.nodes):
            node.global_dofs = np.zeros([self.n_dof_per_node], dtype=int)
            # Assemble DoFs indices to element global_dofs
            for i in range(self.n_dof_per_node):
                element.global_dofs[i + self.n_dof_per_node * n] = (
                    model.dof_connectivity_matrix[i, node.id]
                )
                node.global_dofs[i] = model.dof_connectivity_matrix[i, node.id]

    def apply_prescribed_displacements(self, model: Model, load_case: LoadCase) -> None:
        """
        Add prescribed displacements to the global displacement vector for a given load case.

        This method is responsible for applying known displacement, into the global
        displacement vector, ensuring these displacements are applied only to DoFs that
        are fixed and not to free DoFs or DoFs constrained by springs.

        :param model: A reference to an instance of the :class:`Model` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        for node, prescribed_displacement in load_case.prescribed_displacements.items():
            # get node id
            node_idx = node.id
            # loop through all active dofs
            for i, dof in enumerate(self.active_dofs):
                # get index of dof
                idx = model.dof_connectivity_matrix[i, node_idx]

                # save prescribed displacement to the global vector of displacements
                if idx >= (model.neq_free + model.neq_spring):
                    load_case.u_global[idx] = (
                        prescribed_displacement.prescribed_displacements[dof.index]
                    )

    def assemble_spring_stiffness(self, model: Model) -> None:
        """
        Assembles spring stiffness coefficients into the global stiffness matrix.

        This method iterates through each node in the model, examining the `spring_stiffness`
        attribute of each node to determine if spring stiffness coefficients should be added
        to the global stiffness matrix. The method ensures that spring stiffness is only added
        for DoFs that are constrained by springs.

        :param model: A reference to an instance of the :class:`Model` class.
        """
        # TODO: Move to Model class.
        model.spring_stiffness_global = np.zeros([1, model.neq_spring])

        for n, node in enumerate(model.nodes):
            # TODO: Check if node.fixity == 'spring'
            if node.spring_stiffness:
                for i, dof in enumerate(self.active_dofs):
                    idx = model.dof_connectivity_matrix[i, n]

                    if (
                        (idx > model.neq_free)
                        and (idx <= model.neq_free + model.neq_spring)
                        and (node.spring_stiffness[dof.index] != 0)
                    ):
                        model.k_global[idx, idx] += node.spring_stiffness[dof.index]
                        model.spring_stiffness_global[idx - model.neq_free] = (
                            node.spring_stiffness[dof.index]
                        )
