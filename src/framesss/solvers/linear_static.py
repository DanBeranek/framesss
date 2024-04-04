from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy as sp  # type: ignore[import-untyped]

from framesss.solvers.solver import Solver

if TYPE_CHECKING:
    from framesss.fea.models.model import Model
    from framesss.pre.cases import LoadCase


class LinearStaticSolver(Solver):
    """Subclass of the :class:`Solver` class for linear static solver."""

    def __init__(self, model: Model) -> None:
        """Init the LinearStaticSolver object."""
        super().__init__(model)

    def solve_load_case(self, load_case: LoadCase) -> None:
        """
        Solve the system of equilibrium equations for a specific load case.

        The method divides the stiffness matrix (k_global), the force vector (f_global),
        and the displacement vector (u_global) based on the DoFs categorizations:

          - f: free DoFs (numbered first), natural B.C. (unknown)
          - s: spring DoFs (numbered after free DoFs), natural B.C. (unknown)
          - c: fixed DoFs with known displacements (numbered last), essential B.C. (known)

        Spring DoFs have unknown displacement values, so they are treated as free DoFs on the next steps.

        The equilibrium system is partitioned into sub-matrices and vectors corresponding to these categories:
        [ k_ff k_fc ] * [ u_f ] = [ f_f ]
        [ k_cf k_cc ] * [ u_c ] = [ f_c ]

        The method then solves for the unknown displacements and updates the global displacement and force
        vectors with the solved values and computed reactions.

        :param load_case: The load case being solved, including applied loads and predefined displacements.
        :raises ValueError: If the free-free global stiffness matrix is singular (unstable), indicating an
                            ill-conditioned system that cannot be reliably solved.
        """
        k_global = sp.sparse.lil_matrix(self.model.k_global)

        neq_free = self.model.neq_free
        neq_spring = self.model.neq_spring
        neq = self.model.neq

        # Assemble free-free global matrix
        k_ff = k_global[: (neq_free + neq_spring), : (neq_free + neq_spring)].tocsr()

        # Check for stable k_ff global matrix by verifying its determinant (a very low
        # determinant indicates that the matrix is badly conditioned and may be singular)
        if np.linalg.det(k_ff.todense()) < 10.0e-12:
            raise ValueError("Singular stiffness matrix.")

        # Partition system of equations
        k_fc = k_global[: (neq_free + neq_spring), (neq_free + neq_spring) : neq]
        k_cf = k_global[(neq_free + neq_spring) : neq, : (neq_free + neq_spring)]
        k_cc = k_global[(neq_free + neq_spring) : neq, (neq_free + neq_spring) : neq]

        f_f = load_case.f_global[: (neq_free + neq_spring)]
        f_c = load_case.f_global[(neq_free + neq_spring) :]

        u_c = load_case.u_global[(neq_free + neq_spring) :]

        u_f = sp.sparse.linalg.spsolve(k_ff, f_f - k_fc @ u_c)

        # Recover forcing unknown values (reactions) at essential B.C. It is assumed that the
        # f_c vector currently stores combined nodal loads applied directly to fixed DoFs.
        # Superimpose computed reaction values to combined nodal loads, with inverse direction,
        # that were applied directly to fixed DoFs
        f_c = -f_c + k_cf @ u_f + k_cc @ u_c

        # Initialise spring reaction vector
        f_s = np.zeros(self.model.neq_spring)

        for i in range(self.model.neq_spring):
            f_s[i] = (
                -self.model.spring_stiffness_global[i] * u_f[self.model.neq_free + i]
            )

        f_f[self.model.neq_free : self.model.neq_free + self.model.neq_spring] = f_s

        # Reconstruct the global force vector f_global and the global displacement vector u_global
        load_case.u_global = np.concatenate([u_f, u_c])
        load_case.f_global = np.concatenate([f_f, f_c])

    def solve(self, verbose: bool = False) -> None:
        """
        Execute the finite element analysis (FEA) process for the model.

        This method systematically processes the structural model, performing steps from discretization
        of members to the calculation of internal forces and displacements for each load case.
        It supports verbose output to monitor the progress through these steps.

        :param verbose: If True, detailed progress of each analysis step is printed to the console.
        """
        steps_init = 6
        steps_cases = len(self.model.load_cases) * 7
        steps_combs = len(self.model.load_combinations) * 3
        steps_envelopes = len(self.model.envelopes) * 1
        n_steps = steps_init + steps_cases + steps_combs + steps_envelopes

        if verbose:
            print(f"1/{n_steps} : Preparing fea data...")

        self.model.discretize_members()
        self.model.setup_fictitious_rotation_constraints(True)

        if verbose:
            print(f"2/{n_steps} : Number of equations: {self.model.neq}...")

        if verbose:
            print(
                f"3/{n_steps}"
                f" : Initialization of global stiffness matrix and force vectors..."
            )
        self.init_global_matrices_vectors()

        if verbose:
            print(f"4/{n_steps} : Generating global DoFs numbering matrix...")
        self.model.analysis.setup_dof_numbers(self.model)

        if verbose:
            print(f"5/{n_steps} : Assembling global DoFs numbering matrix...")
        self.model.assemble_dof_indices()
        self.model.assemble_global_dofs()

        if verbose:
            print(f"6/{n_steps} : Assembling global stiffness matrix...")
        self.assemble_global_stiffness_matrix()
        self.model.analysis.assemble_spring_stiffness(self.model)

        for i, load_case in enumerate(self.model.load_cases):
            if verbose:
                print(
                    f"{i*7 + steps_init + 1}/{n_steps}"
                    f" : Applying prescribed displacements for load case: '{load_case.label}..."
                )
            self.model.analysis.apply_prescribed_displacements(self.model, load_case)

            if verbose:
                print(
                    f"{i*7 + steps_init + 2}/{n_steps}"
                    f" : Applying nodal forces for load case: '{load_case.label}'..."
                )
            self.model.analysis.assemble_nodal_loads(self.model, load_case)

            if verbose:
                print(
                    f"{i*7 + steps_init + 3}/{n_steps}"
                    f" : Applying member forces for load case: '{load_case.label}'..."
                )
            load_case.assemble_equivalent_nodal_loads()

            if verbose:
                print(
                    f"{i*7 + steps_init + 3}/{n_steps}"
                    f" : Solving load case: '{load_case.label}'..."
                )
            self.solve_load_case(load_case)

            if verbose:
                print(
                    f"{i*7 + steps_init + 4}/{n_steps}"
                    f" : Computing reactions for load case: '{load_case.label}'..."
                )
            self.save_displacements(load_case)
            self.save_reactions(load_case)

            if verbose:
                print(
                    f"{i*7 + steps_init + 5}/{n_steps}"
                    f" : Computing internal forces for load case: '{load_case.label}'..."
                )
            self.save_element_internal_forces(load_case)
            self.save_member_internal_forces(load_case)

            if verbose:
                print(
                    f"{i*7 + steps_init + 6}/{n_steps}"
                    f" : Computing internal displacements for load case: '{load_case.label}'..."
                )
            self.save_member_internal_displacements(load_case)

            load_case.is_solved = True

        for i, load_combination in enumerate(self.model.load_combinations):
            if verbose:
                print(
                    f"{i*3 + steps_init + steps_cases + 1}/{n_steps}"
                    f" : Computing internal forces for load combination: "
                    f"'{load_combination.label}'..."
                )
            self.save_displacements_combination(load_combination)
            self.save_reactions_combination(load_combination)

            if verbose:
                print(
                    f"{i*3 + steps_init + steps_cases + 2}/{n_steps}"
                    f" : Computing internal forces for load combination: "
                    f"'{load_combination.label}'..."
                )
            self.save_member_internal_forces(load_combination)

            if verbose:
                print(
                    f"{i*3 + steps_init + steps_cases + 3}/{n_steps}"
                    f" : Computing internal displacements for load combination: "
                    f"'{load_combination.label}'..."
                )
            self.save_member_internal_displacements_combination(load_combination)

        for i, envelope in enumerate(self.model.envelopes):
            if verbose:
                print(
                    f"{i*1 + steps_init + steps_cases + steps_combs + 1}/{n_steps}"
                    f" : Computing internal forces for envelope: {envelope.label}..."
                )
            self.save_envelope_internal_forces(envelope)

        if verbose:
            print(f"{n_steps}/{n_steps} : Analysis successfully finished.")
