from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from framesss.fea.boundary_conditions.element_load import DistributedLoad
from framesss.fea.boundary_conditions.element_load import ThermalLoad
from framesss.fea.boundary_conditions.nodal_load import NodalLoad
from framesss.fea.boundary_conditions.prescribed_displacement import (
    PrescribedDisplacement,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from framesss.fea.element_1d import Element1D
    from framesss.fea.node import Node


class LoadCase:
    """
    Represents a specific load case within a finite element analysis model.

    A load case encapsulates all loading conditions and constraints applied to the model
    for a particular analysis scenario. This includes nodal loads, prescribed displacements,
    and distributed and thermal loads on elements. The class facilitates the organization and
    application of these loads and constraints, and stores the results (global force and
    displacement vectors) once the load case has been solved.

    :param label: A unique identifier for the load case.
    :ivar nodal_loads: A dictionary mapping :class:`Node` instances to :class:`NodalLoad`
                       objects, representing loads applied directly to nodes.
    :ivar prescribed_displacements: A dictionary mapping :class:`Node` instances to
                                    :class:`PrescribedDisplacement` objects.
    :ivar element_distributed_loads: A dictionary mapping :class:`Element1D` instances to
                                     :class:`DistributedLoad` objects, specifying loads
                                     distributed along the length of elements.
    :ivar element_thermal_loads: A dictionary mapping :class:`Element1D` instances to
                                 :class:`ThermalLoad` objects, specifying thermal loads
                                 affecting elements.
    :ivar f_global: The global force vector for this load case, which is calculated based
                    on the applied loads. Initialized as None and updated upon solving the load case.
    :ivar u_global: The global displacement vector resulting from the applied loads and constraints,
                    determined after the load case is solved. Initialized as None.
    :ivar is_solved: A boolean flag indicating whether the load case has been analyzed and solved.
    """

    def __init__(self, label: str) -> None:
        """Init the LoadCase class."""
        self.label = label
        self.nodal_loads: dict[Node, NodalLoad] = {}
        self.prescribed_displacements: dict[Node, PrescribedDisplacement] = {}
        self.element_distributed_loads: dict[Element1D, DistributedLoad] = {}
        self.element_thermal_loads: dict[Element1D, ThermalLoad] = {}
        self.f_global: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.u_global: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
        self.is_solved: bool = False

    def __repr__(self) -> str:
        """Return a string representation of LoadCase object."""
        return f"{self.__class__.__name__}({self.label})"

    def assemble_equivalent_nodal_loads(self) -> None:
        """
        Add member equivalent nodal loads to global force vector.

        Assembles member equivalent nodal load vector (in global system)
        to any term of the global forcing vector, including the terms that
        correspond to constrained DoFs.
        """
        for elem, distributed_load in self.element_distributed_loads.items():
            # get global dofs
            dofs = elem.global_dofs
            # get member equivalent nodal loads
            feg = distributed_load.get_equivalent_nodal_actions()
            # assemble equivalent nodal loads to global force vector
            self.f_global[dofs] += feg

        for elem, thermal_load in self.element_thermal_loads.items():
            # get global dofs
            dofs = elem.global_dofs
            # get member equivalent nodal loads
            feg = thermal_load.get_equivalent_nodal_actions()
            # assemble equivalent nodal loads to global force vector
            self.f_global[dofs] += feg


class LoadCaseCombination:
    """
    Represent a combination of load cases.

    :param label: A unique identifier for the load combination.
    :ivar load_cases: A dictionary mapping :class:`LoadCase` instances
                        to their scaling factors.
    """

    def __init__(self, label: str, load_cases: dict[LoadCase, float]) -> None:
        """Init the LoadCaseCombination class."""
        self.label = label
        self.load_cases = load_cases

    def __repr__(self) -> str:
        """Return a string representation of LoadCaseCombination object."""
        return f"{self.__class__.__name__}({self.label})"

    def add_load_case(self, load_case: LoadCase, factor: float) -> None:
        """
        Add load case to load combination.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :param factor: The factor for a given load case.
        """
        self.load_cases[load_case] = factor


class NonlinearLoadCaseCombination(LoadCase):
    """Represent a combination of load cases for nonlinear analysis."""

    def __init__(self, label: str, load_cases: dict[LoadCase, float]) -> None:
        """Init the NonlinearLoadCaseCombination class."""
        super().__init__(label)
        self.label = label
        self.load_cases = load_cases

        self.set_loads()

    def __repr__(self) -> str:
        """Return a string representation of NonlinearLoadCaseCombination object."""
        return f"{self.__class__.__name__}({self.label})"

    def add_load_case(self, load_case: LoadCase, factor: float) -> None:
        """
        Add load case to load combination.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :param factor: The factor for a given load case.
        """
        self.load_cases[load_case] = factor

    def set_loads(self) -> None:
        """
        This method must be called after model.discretize() method.
        """
        for load_case, factor in self.load_cases.items():
            for node, lc_nodal_load in load_case.nodal_loads.items():
                if not self.nodal_loads.get(node):
                    self.nodal_loads[node] = NodalLoad()

                co_nodal_load = self.nodal_loads[node]
                co_nodal_load.load_components += lc_nodal_load.load_components * factor

            for node, lc_pd in load_case.prescribed_displacements.items():
                if not self.prescribed_displacements.get(node):
                    self.prescribed_displacements[node] = PrescribedDisplacement()

                co_pd = self.prescribed_displacements[node]
                co_pd.prescribed_displacements += (
                    lc_pd.prescribed_displacements * factor
                )

            for element, lc_edl in load_case.element_distributed_loads.items():
                if not self.element_distributed_loads.get(element):
                    self.element_distributed_loads[element] = DistributedLoad(element)

                co_edl = self.element_distributed_loads[element]
                co_edl.components_local += lc_edl.components_local * factor

            for element, lc_etl in load_case.element_thermal_loads.items():
                if not self.element_thermal_loads.get(element):
                    self.element_thermal_loads[element] = ThermalLoad(element)

                co_etl = self.element_thermal_loads[element]
                co_etl.temperature_gradients += lc_etl.temperature_gradients * factor


class EnvelopeCombination:
    """
    Represent an envelope of load cases and (or) load combinations.

    This class is designed to create an envelope of internal forces.

    :param label: A unique identifier for the envelope combination.
    :ivar cases: A list of :class:`LoadCase` and :class:`LoadCombination`.
    """

    def __init__(
        self,
        label: str,
        cases: list[LoadCase | LoadCaseCombination | NonlinearLoadCaseCombination],
    ) -> None:
        """Init the LoadCaseCombination class."""
        self.label = label
        self.cases = cases

    def __repr__(self) -> str:
        """Return a string representation of EnvelopeCombination object."""
        return f"{self.__class__.__name__}({self.label})"

    def add_case(
        self, case: LoadCase | LoadCaseCombination | NonlinearLoadCaseCombination
    ) -> None:
        """
        Add load case or load combination to load combination.

        :param case: A reference to an instance of the :class:`LoadCase` and :class:`LoadCombination`.
        """
        self.cases.append(case)
