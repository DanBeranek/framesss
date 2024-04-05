from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import numpy.typing as npt

    from framesss.fea.models.model import Model
    from framesss.pre.cases import EnvelopeCombination
    from framesss.pre.cases import LoadCase
    from framesss.pre.cases import LoadCaseCombination


class Solver(ABC):
    """
    Class for a finite element solver.

    An abstract base class that outlines the structure and requirements of solver
    algorithms for finite element analysis (FEA).

    Derived solver classes must implement specific methods to carry out the solution
    process for an FEA model. These implementations can vary significantly.

    :param model: A reference to an instance of the :class:`Model` class. This model
                  contains all the data necessary for the solver to perform the analysis.

    The initialization of a Solver instance requires a reference to a fully defined FEA model,
    ensuring that all necessary data for solving the structural problem is accessible.
    """

    def __init__(self, model: Model) -> None:
        """Init the Solver object."""
        self.model = model

    def __repr__(self) -> str:
        """Return class representation of Class object."""
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__})"

    def init_global_matrices_vectors(self) -> None:
        """
        Initialize global matrices and vectors for the model.

        This method sets up the global stiffness matrix and initializes global force and displacement
        vectors for each load case within the model.

        The global stiffness matrix is initialized as a sparse matrix to optimize memory usage and
        computational efficiency, especially important for large-scale structural analysis problems.
        Global force and displacement vectors for each load case are initialized with zeros, ready to
        be populated with specific values as the analysis progresses.
        """
        self.model.k_global = sp.sparse.coo_matrix((self.model.neq, self.model.neq))

        for case in self.model.load_cases:
            case.f_global = np.zeros(self.model.neq)
            case.u_global = np.zeros(self.model.neq)

    def assemble_global_stiffness_matrix(self) -> None:
        """
        Assembles the global stiffness matrix for the entire model using the sparse COO format.

        This method iterates over all elements in the model, extracting each element's stiffness matrix and
        its associated global degrees of freedom (DoFs). These matrices are then combined into a single global
        stiffness matrix. The sparse COO format is utilized to efficiently store and manage the non-zero values
        of the stiffness matrix.
        """
        row: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        col: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        data: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)

        for element in self.model.elements:
            dofs = element.global_dofs
            n_dofs = len(dofs)

            keg = element.get_element_global_stiffness_matrix()

            r = np.repeat(dofs, n_dofs)
            c = np.tile(dofs, n_dofs)

            k = keg.flatten()

            row = np.hstack((row, r))
            col = np.hstack((col, c))
            data = np.hstack((data, k))

        k_global = sp.sparse.coo_matrix(
            (data, (row, col)), shape=(self.model.neq, self.model.neq)
        )

        # TODO: Assemble spring stiffness here

        self.model.k_global = k_global

    def save_element_internal_forces(self, load_case: LoadCase) -> None:
        """
        Calculate and save the internal forces for each element in the model under a specified load case.

        For each element in the model, the method:
          - Retrieves the internal actions derived from global analysis results.
          - Adds contributions from fixed end forces due to distributed loads and thermal effects, if present
            for the element under the given load case.
          - Assembles the calculated internal forces into appropriate arrays or structures for further analysis.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        for elem in self.model.elements:

            # Get member internal forces from global analysis (from nodal displacements and rotations)
            fel = elem.get_element_internal_actions(load_case)

            # Compute member internal forces from local analysis (fixed end forces)
            if distributed_load := load_case.element_distributed_loads.get(elem):
                fel += self.model.analysis.get_fixed_end_forces(distributed_load)

            if thermal_load := load_case.element_thermal_loads.get(elem):
                fel += self.model.analysis.get_fixed_end_forces(thermal_load)

            # Assemble member internal force arrays
            self.model.analysis.assemble_internal_forces(elem, load_case, fel)

    def save_member_internal_forces(self, case: LoadCase | LoadCaseCombination) -> None:
        """
        Save the internal forces for each member in the model under a specified load case.

        This method iterates over all members in the model and invokes a process to calculate and store
        the internal stresses (forces and moments) experienced by each member due to the applied loads
        in the given load case. The calculation is performed by the `save_internal_stresses_on_member`
        method of the analysis model, which takes into account both global and local effects to
        accurately determine the stresses along each member.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        """
        for member in self.model.members:
            self.model.analysis.save_internal_stresses(member, case)

    def save_envelope_internal_forces(self, envelope: EnvelopeCombination) -> None:
        """
        Save the envelope of internal forces for each member.

        This method iterates over all members in the model and invokes a process to calculate and store
        the envelope of internal stresses (forces and moments) experienced by each member due to the applied
        loads.
        """
        for member in self.model.members:
            self.model.analysis.save_envelope_stresses(member, envelope)

    def save_member_internal_displacements(self, load_case: LoadCase) -> None:
        """
        Save the displacements for each member in the model under a specified load case.

        This method iterates over all members in the model and invokes a process to calculate and store
        the displacements (translations and rotations) experienced by each member due to the applied loads
        in the given load case. The calculation is performed by the `save_internal_displacements_on_member`
        method of the analysis model, which takes into account both global and local effects to
        accurately determine the stresses along each member.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        for member in self.model.members:
            self.model.analysis.save_internal_displacements_on_member(member, load_case)

    def save_member_internal_displacements_combination(
        self, load_combination: LoadCaseCombination
    ) -> None:
        """
        Save the displacements for each member in the model under a specified load case.

        This method iterates over all members in the model and invokes a process to calculate and store
        the displacements (translations and rotations) experienced by each member due to the applied loads
        in the given load case. The calculation is performed by the `save_internal_displacements_on_member`
        method of the analysis model, which takes into account both global and local effects to
        accurately determine the stresses along each member.

        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        """
        for member in self.model.members:
            self.model.analysis.save_internal_displacements_on_member_combination(
                member, load_combination
            )

    def save_reactions(self, load_case: LoadCase) -> None:
        """
        Save the reaction forces and moments for each node in the model under a specified load case.

        This method iterates over all nodes in the model, retrieving and storing the reaction forces
        and moments resulting from the applied loads and constraints defined by the load case.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        for node in self.model.nodes:
            self.model.analysis.save_reactions(node, load_case)

    # TODO: docstring
    def save_reactions_combination(self, load_combination: LoadCaseCombination) -> None:
        """
        Save the reaction forces and moments for each node in the model under a specified load case.

        This method iterates over all nodes in the model, retrieving and storing the reaction forces
        and moments resulting from the applied loads and constraints defined by the load case.

        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        """
        for node in self.model.nodes:
            self.model.analysis.save_reactions_combination(node, load_combination)

    def save_displacements(self, load_case: LoadCase) -> None:
        """
        Save the displacements for each node in the model under a specified load case.

        This method iterates over all nodes in the model, retrieving and storing the displacements
        (translations and rotations) resulting from the applied loads and constraints defined by the load case.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        for node in self.model.nodes:
            self.model.analysis.save_displacements(node, load_case)

    # TODO: docstring
    def save_displacements_combination(
        self, load_combination: LoadCaseCombination
    ) -> None:
        """
        Save the displacements for each node in the model under a specified load case.

        This method iterates over all nodes in the model, retrieving and storing the displacements
        (translations and rotations) resulting from the applied loads and constraints defined by the load case.

        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        """
        for node in self.model.nodes:
            self.model.analysis.save_displacements_combination(node, load_combination)

    @abstractmethod
    def solve_load_case(self, load_case: LoadCase) -> None:
        """Solve the system of equilibrium equations for a specific load case."""
        pass

    @abstractmethod
    def solve(self, verbose: bool) -> None:
        """
        Abstract method to solve the FEA problem.

        This method serves as a placeholder for solver algorithms that compute the solution to structural
        analysis problems. Concrete implementations of this method in subclasses should define the specific
        steps and algorithms used to solve the FEA problem.

        :param verbose: If True, detailed progress of each analysis step is printed to the console.

        Implementing this method in a subclass involves utilizing the global stiffness matrix, load vectors, and any
        other necessary structural model data to calculate the response of the structure under the specified load case.
        The implementation must handle the assembly and manipulation of matrices and vectors, application of boundary
        conditions, and ultimately, the solution of the system of equations to find displacements and reactions.

        :raise NotImplementedError: If called directly from the abstract base class, indicating that concrete
                                    subclasses are required to provide an implementation of this method.
        """
        raise NotImplementedError(
            "The 'solve' method must be implemented in concrete subclasses."
        )
