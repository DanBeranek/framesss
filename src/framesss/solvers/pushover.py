from __future__ import annotations

from typing import TYPE_CHECKING

from copy import copy

import numpy as np
import scipy as sp  # type: ignore[import-untyped]

from framesss.solvers.solver import Solver
from framesss.utils import is_invertible
from framesss.errors import SingularMatrixError

if TYPE_CHECKING:
    from framesss.fea.models.model import Model
    from framesss.pre.cases import NonlinearLoadCaseCombination


class PushoverSolver(Solver):
    """Subclass of the :class:`Solver` class for pushover analysis."""

    def __init__(self, model: Model) -> None:
        """Init the LinearStaticSolver object."""
        super().__init__(model)

    def solve_load_case(
        self,
        combination: NonlinearLoadCaseCombination,
        n_time_steps: int = 20,
        modulus_type: str = "tangent",
    ) -> None:
        """
        Solve the system of equilibrium equations for a specific load case.

        :param combination: The nonlinear load case being solved, including applied
                            loads and predefined displacements.
        :param n_time_steps: The number of time steps to solve the load case.
        :param modulus_type: The type of modulus to use for the analysis.
                             Can be either 'tangent' or 'secant'.
        """
        neq_free = self.model.neq_free
        neq_spring = self.model.neq_spring
        neq = self.model.neq

        # store global force vector
        f_global = copy(combination.f_global)

        # print(f'Solving combination {combination}')

        for i, coefficient in enumerate(np.linspace(0, 1, n_time_steps + 1)[1:]):
            # print(f'Load step {i+1}')
            # print(f'{coefficient=}')
            if i == 0:
                self.assemble_global_stiffness_matrix()
            else:
                self.assemble_global_stiffness_matrix(
                    nonlinear_combination=combination, modulus_type=modulus_type
                )

            k_global = sp.sparse.lil_matrix(self.model.k_global)

            # Assemble free-free global matrix
            k_ff = k_global[
                : (neq_free + neq_spring), : (neq_free + neq_spring)
            ].tocsr()

            # Check for stable k_ff global matrix by verifying its determinant (a very low
            # determinant indicates that the matrix is badly conditioned and may be singular)
            if not is_invertible(k_ff.todense()):
                raise SingularMatrixError(
                    f"Singular stiffness matrix. Determinant: {np.linalg.det(k_ff.todense())}",
                    matrix=k_ff,
                )

            # Partition system of equations
            k_fc = k_global[: (neq_free + neq_spring), (neq_free + neq_spring) : neq]
            k_cf = k_global[(neq_free + neq_spring) : neq, : (neq_free + neq_spring)]
            k_cc = k_global[
                (neq_free + neq_spring) : neq, (neq_free + neq_spring) : neq
            ]

            f_f = f_global[: (neq_free + neq_spring)] * coefficient
            f_c = f_global[(neq_free + neq_spring) :] * coefficient

            u_c = combination.u_global[(neq_free + neq_spring) :]

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
                    -self.model.spring_stiffness_global[i]
                    * u_f[self.model.neq_free + i]
                )

            f_f[self.model.neq_free : self.model.neq_free + self.model.neq_spring] = f_s

            # Reconstruct the global force vector f_global and the global displacement vector u_global
            combination.u_global = np.concatenate([u_f, u_c])
            combination.f_global = np.concatenate([f_f, f_c])

            for element in self.model.elements:
                self.model.analysis.save_curvatures_xz(
                    element=element, combination=combination
                )

        print(f"{combination.label} solved")

    def solve(
        self,
        verbose: bool = False,
        max_element_length: float = 0.2,
        n_time_steps: int = 20,
        modulus_type: str = "tangent",
    ) -> None:
        """
        Solve the pushover analysis.
        """
        steps_init = 6
        steps_cases = len(self.model.load_cases) * 7
        steps_combs = len(self.model.load_combinations) * 3
        steps_envelopes = len(self.model.envelopes) * 1
        n_steps = steps_init + steps_cases + steps_combs + steps_envelopes

        if verbose:
            print(f"1/{n_steps} : Preparing fea data...")

        self.model.discretize_members(max_element_length=max_element_length)
        self.model.setup_fictitious_rotation_constraints(fict=True)

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

        for i, case in enumerate(self.model.nonlinear_load_combinations):
            case.set_loads()
            # TODO: Apply prescribed displacements
            # if verbose:
            #     print(
            #         f"{i*7 + steps_init + 1}/{n_steps}"
            #         f" : Applying prescribed displacements for nonlinear combination: '{case.label}..."
            #     )
            # self.model.analysis.apply_prescribed_displacements(self.model, case)

            if verbose:
                print(
                    f"{i*7 + steps_init + 2}/{n_steps}"
                    f" : Applying nodal forces for nonlinear combination: '{case.label}'..."
                )
            self.model.analysis.assemble_nodal_loads_nonlinear_combination(
                self.model, case
            )

            if verbose:
                print(
                    f"{i*7 + steps_init + 3}/{n_steps}"
                    f" : Applying member forces for nonlinear combination: '{case.label}'..."
                )
            case.assemble_equivalent_nodal_loads()

            if verbose:
                print(
                    f"{i*7 + steps_init + 3}/{n_steps}"
                    f" : Solving nonlinear combination: '{case.label}'..."
                )
            self.solve_load_case(
                combination=case,
                n_time_steps=n_time_steps,
                modulus_type=modulus_type,
            )

            if verbose:
                print(
                    f"{i*7 + steps_init + 4}/{n_steps}"
                    f" : Computing reactions for nonlinear combination: '{case.label}'..."
                )
            self.save_displacements(case)
            self.save_reactions(case)

            if verbose:
                print(
                    f"{i*7 + steps_init + 5}/{n_steps}"
                    f" : Computing internal forces for nonlinear combination: '{case.label}'..."
                )
            self.save_element_internal_forces(case)
            self.save_member_internal_forces(case)

            if verbose:
                print(
                    f"{i*7 + steps_init + 6}/{n_steps}"
                    f" : Computing internal displacements for nonlinear combination: '{case.label}'..."
                )
            self.save_member_internal_displacements(case)

            case.is_solved = True

        for i, envelope in enumerate(self.model.envelopes):
            if verbose:
                print(
                    f"{i*1 + steps_init + steps_cases + steps_combs + 1}/{n_steps}"
                    f" : Computing internal forces for envelope: {envelope.label}..."
                )
            self.save_envelope_internal_forces(envelope)
            self.save_member_internal_displacements_envelope(envelope)
            self.save_reactions_envelope(envelope)

        if verbose:
            print(f"{n_steps}/{n_steps} : Analysis successfully finished.")
