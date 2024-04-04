from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from framesss.enums import BeamConnection
from framesss.enums import Element1DType

if TYPE_CHECKING:
    import numpy.typing as npt

    from framesss.fea.element_1d import Element1D


class ElementLoad(ABC):
    """
    Abstract base class for defining different type of loads on an elements.

    The ElementLoad class is responsible for computation of element fixed end forces,
    equivalent nodal loads, and internal displacements. Each ElementLoad
    object is associated with a specific Element1D instance, emphasizing the direct
    relationship between a member's physical properties and the loads it experiences.
    Subclasses of ElementLoad should implement specific types of loads and their
    corresponding effects on the member.

    :param element: A reference to an instance of the :class:`Element1D` class.
    """

    def __init__(self, element: Element1D) -> None:
        """Init the ElementLoad class."""
        self.element = element

    def __repr__(self) -> str:
        """Return a string representation of the ElementLoad object."""
        element_repr = repr(self.element)
        return f"{self.__class__.__name__}(element={element_repr})"

    def get_equivalent_nodal_actions(self) -> npt.NDArray[np.float64]:
        """
        Return the equivalent nodal actions for an element in the global coordinate system.

        This method first computes the fixed end forces (FEF) for the element in its local
        coordinate system. It then transforms these forces into the equivalent nodal actions
        (ENA) in the global coordinate system.

        :return: The equivalent nodal actions for an element in the global coordinate system.
        """
        fef_local = self.element.member.analysis.get_fixed_end_forces(self)

        ena_global = -self.element.member.transformation_matrix.T @ fef_local

        return ena_global

    @abstractmethod
    def get_axial_fixed_end_forces(self) -> npt.NDArray[np.float64]:
        """Return axial fixed end force vector."""
        pass

    @abstractmethod
    def get_flexural_xy_fixed_end_forces(self) -> npt.NDArray[np.float64]:
        """Return flexural fixed end force vector in local xy-plane."""
        pass

    @abstractmethod
    def get_flexural_xz_fixed_end_forces(self) -> npt.NDArray[np.float64]:
        """Return flexural fixed end force vector in local xz-plane."""
        pass

    @abstractmethod
    def get_axial_displacements(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return axial displacements at specified positions."""
        pass

    @abstractmethod
    def get_flexural_xy_displacements(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return displacements in local xy-plane at specified positions."""
        pass

    @abstractmethod
    def get_flexural_xz_displacements(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return displacements in local xz-plane at specified positions."""
        pass


class DistributedLoad(ElementLoad):
    """
    Class representing distributed load acting on an element.

    :param element: A reference to an instance of the :class:`Element1D` class.
    :ivar components_local: An array of the force components in local coordinate
                            system at both nodes [fx1, fy1, fz1, fx2, fy2, fz2].
    """

    def __init__(self, element: Element1D) -> None:
        """Init DistributedLoad object."""
        super().__init__(element)

        self.components_local: npt.NDArray[np.float64] = np.zeros(6, dtype=np.float64)

    def __repr__(self) -> str:
        """Return a string representation of the DistributedLoad object."""
        element_repr = repr(self.element)
        components_str = ", ".join(f"{comp:.2f}" for comp in self.components_local)
        return (
            f"{self.__class__.__name__}(element={element_repr}, "
            f"components_local=[{components_str}])"
        )

    def get_axial_fixed_end_forces(self) -> npt.NDArray[np.float64]:
        """
        Return axial fixed end force vector.

        The forces are calculated and returned in the local coordinate system of the element.

        :return: A 1x2 array representing the axial fixed end forces at the start and end nodes.
        """
        # Initialize axial load values at end nodes
        fx_start = self.components_local[0]
        fx_end = self.components_local[3]

        # Check if axial load is not null over member
        if fx_start or fx_end:
            L = self.element.length

            # Separate uniform portion from linear partition of axial load
            fx_uniform = fx_start
            fx_linear = fx_end - fx_start

            # Calculate fixed end force vector
            fef_axial = np.array(
                [
                    -(fx_uniform * L / 2 + fx_linear * L / 6),
                    -(fx_uniform * L / 2 + fx_linear * L / 3),
                ]
            )

        else:
            fef_axial = np.zeros(2)

        return fef_axial

    def get_flexural_xy_fixed_end_forces(self) -> npt.NDArray[np.float64]:
        """
        Return flexural fixed end force vector in local xy-plane.

        :return: A 4x1 array representing the flexural fixed end forces at the start and end nodes.
        """
        # Initialize transversal load values at nodes
        fy_start = self.components_local[1]
        fy_end = self.components_local[4]

        # Check if transversal load is not null over member
        if fy_start or fy_end:
            # Basic member properties
            L = self.element.length
            L2 = L**2

            # Timoshenko parameter
            if self.element.member.element_type == Element1DType.NAVIER:
                Omega = 0.0

            elif self.element.member.element_type == Element1DType.TIMOSHENKO:
                EI = self.element.member.section.EIz
                GA = self.element.member.section.GAy

                Omega = EI / (GA * L2)
            else:
                raise AttributeError(
                    f"Wrong element type: {self.element.member.element_type}"
                )

            # Auxiliary parameters
            mu = 1 + 12 * Omega
            lamb = 1 + 3 * Omega
            zeta = 1 + 40 * Omega / 3
            xi = 1 + 5 * Omega
            eta = 1 + 15 * Omega
            vartheta = 1 + 4 * Omega
            psi = 1 + 12 * Omega / 5
            varpi = 1 + 20 * Omega / 9
            epsilon = 1 + 80 * Omega / 7
            varrho = 1 + 10 * Omega
            upsilon = 1 + 5 * Omega / 2
            varsigma = 1 + 40 * Omega / 11

            # Separate uniform portion from linear portion of transversal load
            fy_uniform = fy_start
            fy_linear = fy_end - fy_start

            # Calculate fixed end force vector
            if (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                fef_flex = np.array(
                    [
                        -(fy_uniform * L / 2 + fy_linear * 3 * L * zeta / (20 * mu)),
                        -(fy_uniform * L2 / 12 + fy_linear * L2 * eta / (30 * mu)),
                        -(fy_uniform * L / 2 + fy_linear * 7 * L * epsilon / (20 * mu)),
                        fy_uniform * L2 / 12 + fy_linear * L2 * varrho / (20 * mu),
                    ]
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                fef_flex = np.array(
                    [
                        -(
                            fy_uniform * 3 * L * vartheta / (8 * lamb)
                            + fy_linear * L * xi / (10 * lamb)
                        ),
                        0,
                        -(
                            fy_uniform * 5 * L * psi / (8 * lamb)
                            + fy_linear * 2 * L * upsilon / (5 * lamb)
                        ),
                        fy_uniform * L2 / (8 * lamb) + fy_linear * L2 / (15 * lamb),
                    ]
                )

            elif (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                fef_flex = np.array(
                    [
                        -(
                            fy_uniform * 5 * L * psi / (8 * lamb)
                            + fy_linear * 9 * L * varpi / (40 * lamb)
                        ),
                        -(
                            fy_uniform * L2 / (8 * lamb)
                            + fy_linear * 7 * L2 / (120 * lamb)
                        ),
                        -(
                            fy_uniform * 3 * L * vartheta / (8 * lamb)
                            + fy_linear * 11 * L * varsigma / (40 * lamb)
                        ),
                        0,
                    ]
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                fef_flex = np.array(
                    [
                        -(fy_uniform * L / 2 + fy_linear * L / 6),
                        0,
                        -(fy_uniform * L / 2 + fy_linear * L / 3),
                        0,
                    ]
                )

            else:
                raise AttributeError(
                    f"Wrong hinge type: {self.element.hinge_start}, {self.element.hinge_end}"
                )

        else:
            fef_flex = np.zeros([4])

        return fef_flex

    def get_flexural_xz_fixed_end_forces(self) -> npt.NDArray[np.float64]:
        """Return flexural fixed end force vector in local xz-plane.

        :return: A 4x1 array representing the flexural fixed end forces at the start and end nodes.
        """
        # Initialize transversal load values at nodes
        fz_start = self.components_local[2]
        fz_end = self.components_local[5]

        # Check if transversal load is not null over member
        if fz_start or fz_end:
            # Basic member properties
            L = self.element.length
            L2 = L**2

            # Timoshenko parameter
            if self.element.member.element_type == Element1DType.NAVIER:
                Omega = 0.0

            elif self.element.member.element_type == Element1DType.TIMOSHENKO:
                EI = self.element.member.section.EIy
                GA = self.element.member.section.GAz

                Omega = EI / (GA * L2)

            else:
                raise ValueError(
                    f"Unknown member type: {self.element.member.element_type}."
                )

            # Auxiliary parameters
            mu = 1 + 12 * Omega
            lamb = 1 + 3 * Omega
            zeta = 1 + 40 * Omega / 3
            xi = 1 + 5 * Omega
            eta = 1 + 15 * Omega
            vartheta = 1 + 4 * Omega
            psi = 1 + 12 * Omega / 5
            varpi = 1 + 20 * Omega / 9
            epsilon = 1 + 80 * Omega / 7
            varrho = 1 + 10 * Omega
            upsilon = 1 + 5 * Omega / 2
            varsigma = 1 + 40 * Omega / 11

            # Separate uniform portion from linear portion of transversal load
            fz_uniform = fz_start
            fz_linear = fz_end - fz_start

            # Calculate fixed end force vector
            if (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                fef_flex = np.array(
                    [
                        -(fz_uniform * L / 2 + fz_linear * 3 * L * zeta / (20 * mu)),
                        fz_uniform * L2 / 12 + fz_linear * L2 * eta / (30 * mu),
                        -(fz_uniform * L / 2 + fz_linear * 7 * L * epsilon / (20 * mu)),
                        -(fz_uniform * L2 / 12 + fz_linear * L2 * varrho / (20 * mu)),
                    ]
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                fef_flex = np.array(
                    [
                        -(
                            fz_uniform * 3 * L * vartheta / (8 * lamb)
                            + fz_linear * L * xi / (10 * lamb)
                        ),
                        0,
                        -(
                            fz_uniform * 5 * L * psi / (8 * lamb)
                            + fz_linear * 2 * L * upsilon / (5 * lamb)
                        ),
                        -(fz_uniform * L2 / (8 * lamb) + fz_linear * L2 / (15 * lamb)),
                    ]
                )

            elif (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                fef_flex = np.array(
                    [
                        -(
                            fz_uniform * 5 * L * psi / (8 * lamb)
                            + fz_linear * 9 * L * varpi / (40 * lamb)
                        ),
                        fz_uniform * L2 / (8 * lamb)
                        + fz_linear * 7 * L2 / (120 * lamb),
                        -(
                            fz_uniform * 3 * L * vartheta / (8 * lamb)
                            + fz_linear * 11 * L * varsigma / (40 * lamb)
                        ),
                        0,
                    ]
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                fef_flex = np.array(
                    [
                        -(fz_uniform * L / 2 + fz_linear * L / 6),
                        0,
                        -(fz_uniform * L / 2 + fz_linear * L / 3),
                        0,
                    ]
                )

            else:
                raise AttributeError(
                    f"Wrong hinge type: {self.element.hinge_start}, {self.element.hinge_end}"
                )

        else:
            fef_flex = np.zeros([4])

        return fef_flex

    def get_axial_displacements(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Return axial displacements at specified positions.

        :param x: An array of positions along the member's local x-axis at which
                  the axial displacement is to be computed.
        :return: An array of axial displacements at the `x` positions.
        """
        # Initialize axial load values at end nodes
        fx_start = self.components_local[0]
        fx_end = self.components_local[3]

        # Check if transversal load is not null over member
        if fx_start or fx_end:
            EA = self.element.member.section.EA
            L = self.element.length

            # Separate uniform portion from linear partition of axial load
            fx_uniform = fx_start
            fx_linear = fx_end - fx_start

            # Calculate axial displacements from uniform and from linear axial load portion
            u_uniform = fx_uniform / EA * (L * x / 2 - x**2 / 2)
            u_linear = fx_linear / EA * (L * x / 6 - x**3 / (6 * L))

            u = u_uniform + u_linear

        else:
            u = np.zeros(x.shape)

        return u  # type: ignore[no-any-return]

    def get_flexural_xy_displacements(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Return displacements in local xy-plane at specified positions.

        :param x: An array of positions along the member's local x-axis at which
                  the flexural displacement is to be computed.
        :return: An array of transversal displacements at the `x` positions.
        """
        # Initialize transversal load value at end nodes
        fy_start = self.components_local[1]
        fy_end = self.components_local[4]

        # Check if transversal load is not null over the member
        if fy_start or fy_end:
            # Basic member properties
            EI = self.element.member.section.EIz
            L = self.element.length

            L2 = L * L
            L3 = L2 * L

            # Timoshenko parameter
            if self.element.member.element_type == Element1DType.NAVIER:
                Omega = 0.0

            elif self.element.member.element_type == Element1DType.TIMOSHENKO:
                GA = self.element.member.section.GAy

                Omega = EI / (GA * L2)

            else:
                raise AttributeError(
                    f"Wrong element type: {self.element.member.element_type}"
                )

            # Auxiliary parameters
            mu = 1 + 12 * Omega
            lamb = 1 + 3 * Omega
            zeta = 1 + 40 * Omega / 3
            xi = 1 + 5 * Omega
            eta = 1 + 15 * Omega
            vartheta = 1 + 4 * Omega
            psi = 1 + 12 * Omega / 5
            varpi = 1 + 20 * Omega / 9

            # Separate uniform portion from linear portion of transversal load
            fy_uniform = fy_start
            fy_linear = fy_end - fy_start

            # Calculate transversal displacements from uniform and from linear load portion
            if (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                v_uniform = (
                    fy_uniform
                    / EI
                    * (
                        L3 * Omega * x / 2
                        + (L2 / 24 - L2 * Omega / 2) * x**2
                        - L * x**3 / 12
                        + x**4 / 24
                    )
                )
                v_linear = (
                    fy_linear
                    / EI
                    * (
                        3 * L3 * Omega * zeta * x / (20 * mu)
                        + L2 * eta * x**2 / (60 * mu)
                        - (L * zeta / (40 * mu) + L * Omega / 6) * x**3
                        + x**5 / (120 * L)
                    )
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                v_uniform = (
                    fy_uniform
                    / EI
                    * (
                        (L3 * mu / (48 * lamb) + 3 * L3 * Omega * vartheta / (8 * lamb))
                        * x
                        - L2 * Omega * x**2 / 2
                        - L * vartheta * x**3 / (16 * lamb)
                        + x**4 / 24
                    )
                )
                v_linear = (
                    fy_linear
                    / EI
                    * (
                        (L3 * eta / (120 * lamb) + L3 * Omega * xi / (10 * lamb)) * x
                        - (L * xi / (60 * lamb) + L * Omega / 6) * x**3
                        + x**5 / (120 * L)
                    )
                )

            elif (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_start == BeamConnection.HINGED_END
            ):
                v_uniform = (
                    fy_uniform
                    / EI
                    * (
                        5 * L3 * Omega * psi * x / (8 * lamb)
                        + (L2 / (16 * lamb) - L2 * Omega / 2) * x**2
                        - 5 * L * psi * x**3 / (48 * lamb)
                        + x**4 / 24
                    )
                )
                v_linear = (
                    fy_linear
                    / EI
                    * (
                        9 * L3 * Omega * varpi * x / (40 * lamb)
                        + 7 * L2 * x**2 / (240 * lamb)
                        - (3 * L * varpi / (80 * lamb) + L * Omega / 6) * x**3
                        + x**5 / (120 * L)
                    )
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                v_uniform = (
                    fy_uniform
                    / EI
                    * (
                        (L3 / 24 + L3 * Omega / 2) * x
                        - L2 * Omega * x**2 / 2
                        - L * x**3 / 12
                        + x**4 / 24
                    )
                )
                v_linear = (
                    fy_linear
                    / EI
                    * (
                        (7 * L3 / 360 + L3 * Omega / 6) * x
                        - (L / 36 + L * Omega / 6) * x**3
                        + x**5 / (120 * L)
                    )
                )

            else:
                raise AttributeError(
                    f"Wrong hinge type: {self.element.hinge_start}, {self.element.hinge_end}"
                )

            v = v_uniform + v_linear

        else:
            v = np.zeros(x.shape)

        return v  # type: ignore[no-any-return]

    def get_flexural_xz_displacements(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Return displacements in local xz-plane at specified positions.

        :param x: An array of positions along the member's local x-axis at which
                  the flexural displacement is to be computed.
        :return: An array of transversal displacements at the `x` positions.
        """
        # Initialize transversal load value at end nodes
        fz_start = self.components_local[2]
        fz_end = self.components_local[5]

        # Check if transversal load is not null over the member
        if fz_start or fz_end:
            # Basic member properties
            EI = self.element.member.section.EIy
            L = self.element.length

            L2 = L * L
            L3 = L2 * L

            # Timoshenko parameter
            if self.element.member.element_type == Element1DType.NAVIER:
                Omega = 0.0

            elif self.element.member.element_type == Element1DType.TIMOSHENKO:
                GA = self.element.member.section.GAz

                Omega = EI / (GA * L2)

            else:
                raise AttributeError(
                    f"Wrong element type: {self.element.member.element_type}"
                )

            # Auxiliary parameters
            mu = 1 + 12 * Omega
            lamb = 1 + 3 * Omega
            zeta = 1 + 40 * Omega / 3
            xi = 1 + 5 * Omega
            eta = 1 + 15 * Omega
            vartheta = 1 + 4 * Omega
            psi = 1 + 12 * Omega / 5
            varpi = 1 + 20 * Omega / 9

            # Separate uniform portion from linear portion of transversal load
            fz_uniform = fz_start
            fz_linear = fz_end - fz_start

            # Calculate transversal displacements from uniform and from linear load portion
            if (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                w_uniform = (
                    fz_uniform
                    / EI
                    * (
                        L3 * Omega * x / 2
                        + (L2 / 24 - L2 * Omega / 2) * x**2
                        - L * x**3 / 12
                        + x**4 / 24
                    )
                )
                w_linear = (
                    fz_linear
                    / EI
                    * (
                        3 * L3 * Omega * zeta * x / (20 * mu)
                        + L2 * eta * x**2 / (60 * mu)
                        - (L * zeta / (40 * mu) + L * Omega / 6) * x**3
                        + x**5 / (120 * L)
                    )
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                w_uniform = (
                    fz_uniform
                    / EI
                    * (
                        (L3 * mu / (48 * lamb) + 3 * L3 * Omega * vartheta / (8 * lamb))
                        * x
                        - L2 * Omega * x**2 / 2
                        - L * vartheta * x**3 / (16 * lamb)
                        + x**4 / 24
                    )
                )
                w_linear = (
                    fz_linear
                    / EI
                    * (
                        (L3 * eta / (120 * lamb) + L3 * Omega * xi / (10 * lamb)) * x
                        - (L * xi / (60 * lamb) + L * Omega / 6) * x**3
                        + x**5 / (120 * L)
                    )
                )

            elif (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                w_uniform = (
                    fz_uniform
                    / EI
                    * (
                        5 * L3 * Omega * psi * x / (8 * lamb)
                        + (L2 / (16 * lamb) - L2 * Omega / 2) * x**2
                        - 5 * L * psi * x**3 / (48 * lamb)
                        + x**4 / 24
                    )
                )
                w_linear = (
                    fz_linear
                    / EI
                    * (
                        9 * L3 * Omega * varpi * x / (40 * lamb)
                        + 7 * L2 * x**2 / (240 * lamb)
                        - (3 * L * varpi / (80 * lamb) + L * Omega / 6) * x**3
                        + x**5 / (120 * L)
                    )
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                w_uniform = (
                    fz_uniform
                    / EI
                    * (
                        (L3 / 24 + L3 * Omega / 2) * x
                        - L2 * Omega * x**2 / 2
                        - L * x**3 / 12
                        + x**4 / 24
                    )
                )
                w_linear = (
                    fz_linear
                    / EI
                    * (
                        (7 * L3 / 360 + L3 * Omega / 6) * x
                        - (L / 36 + L * Omega / 6) * x**3
                        + x**5 / (120 * L)
                    )
                )

            else:
                raise AttributeError(
                    f"Wrong hinge type: {self.element.hinge_start}, {self.element.hinge_end}"
                )

            w = w_uniform + w_linear

        else:
            w = np.zeros(x.shape)

        return w  # type: ignore[no-any-return]


class ThermalLoad(ElementLoad):
    """
    Class representing thermal load acting on an element.

    :param element: A reference to an instance of the :class:`Element1D` class.
    :ivar temperature_gradients: A list of the temperature gradient relative to local axis
                                 [temperature variation on member center of gravity,
                                 temperature gradient relative to local y-axis,
                                 temperature gradient relative to local z-axis]
    """

    def __init__(self, element: Element1D) -> None:
        """Init ThermalLoad class."""
        super().__init__(element)
        self.temperature_gradients = np.zeros(3, dtype=np.float64)

    def __repr__(self) -> str:
        """Return a string representation of ThermalLoad object."""
        element_repr = repr(self.element)
        gradients_str = ", ".join(f"{comp:.2f}" for comp in self.temperature_gradients)
        return (
            f"{self.__class__.__name__}("
            f"element={element_repr}, "
            f"temperature_gradients=[{gradients_str}])"
        )

    def get_axial_fixed_end_forces(self) -> npt.NDArray[np.float64]:
        """
        Return axial fixed end force vector.

        The forces are calculated and returned in the local coordinate system of the element.

        :return: A 1x2 array representing the axial fixed end forces at the start and end nodes.
        """
        # Get temperature variation on member center of gravity
        dtx = self.temperature_gradients[0]

        if dtx:
            EA = self.element.member.section.EA
            alpha = self.element.member.section.material.thermal_expansion_coefficient

            # Calculate fixed end forces
            fef_axial = np.array(
                [
                    EA * alpha * dtx,
                    -EA * alpha * dtx,
                ]
            )

        else:
            return np.zeros(2)

        return fef_axial

    def get_flexural_xy_fixed_end_forces(self) -> npt.NDArray[np.float64]:
        """
        Return flexural fixed end force vector in local xy-plane.

        :return: A 4x1 array representing the flexural fixed end forces at the start and end nodes.
        """
        # Get temperature gradient relative to member local y-axis
        dty = self.temperature_gradients[1]

        if dty:
            # Basic member properties
            alpha = self.element.member.section.material.thermal_expansion_coefficient
            h = self.element.member.section.height_y
            L = self.element.length
            EI = self.element.member.section.EIz

            # Timoshenko parameter
            if self.element.member.element_type == Element1DType.NAVIER:
                Omega = 0.0

            elif self.element.member.element_type == Element1DType.TIMOSHENKO:
                GA = self.element.member.section.GAy

                Omega = EI / (GA * L * L)

            else:
                raise AttributeError(
                    f"Wrong element type: {self.element.member.element_type}"
                )

            # Auxiliary parameter
            lamb = 1 + 3 * Omega

            # Compute unitary dimensionless temperature gradient
            tg = (alpha * dty) / h

            # Calculate fixed end forces vector
            if (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                fef_flex = np.array([0, tg * EI, 0, -tg * EI])

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                fef_flex = np.array(
                    [
                        tg * (-3 * EI / (2 * L * lamb)),
                        0,
                        tg * (3 * EI / (2 * L * lamb)),
                        tg * (-3 * EI / (2 * lamb)),
                    ]
                )

            elif (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                fef_flex = np.array(
                    [
                        tg * (3 * EI / (2 * L * lamb)),
                        tg * (3 * EI / (2 * lamb)),
                        tg * (-3 * EI / (2 * L * lamb)),
                        0,
                    ]
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                fef_flex = np.zeros(4)

            else:
                raise AttributeError(
                    f"Wrong hinge type: {self.element.hinge_start}, {self.element.hinge_end}"
                )

        else:
            fef_flex = np.zeros(4)

        return fef_flex

    def get_flexural_xz_fixed_end_forces(self) -> npt.NDArray[np.float64]:
        """
        Return flexural fixed end force vector in local xz-plane.

        :return: A 4x1 array representing the flexural fixed end forces at the start and end nodes.
        """
        # Get temperature gradient relative to member local y-axis
        dtz = self.temperature_gradients[2]

        if dtz:
            # Basic member properties
            alpha = self.element.member.section.material.thermal_expansion_coefficient
            h = self.element.member.section.height_z
            L = self.element.length
            EI = self.element.member.section.EIy

            # Timoshenko parameter
            if self.element.member.element_type == Element1DType.NAVIER:
                Omega = 0.0

            elif self.element.member.element_type == Element1DType.TIMOSHENKO:

                GA = self.element.member.section.GAz

                Omega = EI / (GA * L * L)

            else:
                raise AttributeError(
                    f"Wrong element type: {self.element.member.element_type}"
                )

            # Auxiliary parameter
            lamb = 1 + 3 * Omega

            # Compute unitary dimensionless temperature gradient
            tg = (alpha * dtz) / h

            # Calculate fixed end forces vector
            if (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                fef_flex = np.array([0, -tg * EI, 0, tg * EI])

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                fef_flex = np.array(
                    [
                        tg * (-3 * EI / (2 * L * lamb)),
                        0,
                        tg * (3 * EI / (2 * L * lamb)),
                        tg * (3 * EI / (2 * lamb)),
                    ]
                )

            elif (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                fef_flex = np.array(
                    [
                        tg * (3 * EI / (2 * L * lamb)),
                        tg * (-3 * EI / (2 * lamb)),
                        tg * (-3 * EI / (2 * L * lamb)),
                        0,
                    ]
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                fef_flex = np.zeros(4)

            else:
                raise AttributeError(
                    f"Wrong hinge type: {self.element.hinge_start}, {self.element.hinge_end}"
                )

        else:
            fef_flex = np.zeros(4)

        return fef_flex

    def get_axial_displacements(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Return axial displacements at specified positions.

        :param x: An array of positions along the member's local x-axis at which
                  the axial displacement is to be computed.
        :return: An array of axial displacements at the `x` positions.
        """
        # Axial displacement is null when an element with fixed ends is
        # subjected to a temperature variation
        u = np.zeros(x.shape)
        return u

    def get_flexural_xy_displacements(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Return displacements in local xy-plane at specified positions.

        :param x: An array of positions along the member's local x-axis at which
                  the flexural displacement is to be computed.
        :return: An array of transversal displacements at the `x` positions.
        """
        # Get temperature gradient relative to member local y-axis
        dty = self.temperature_gradients[1]

        # Check if temperature gradient is not null
        if dty:
            # Basic member properties
            alpha = self.element.member.section.material.thermal_expansion_coefficient
            h = self.element.member.section.height_y
            L = self.element.length

            # Timoshenko parameter
            if self.element.member.element_type == Element1DType.NAVIER:
                Omega = 0.0

            elif self.element.member.element_type == Element1DType.TIMOSHENKO:
                EI = self.element.member.section.EIz
                GA = self.element.member.section.GAy

                Omega = EI / (GA * L * L)

            else:
                raise AttributeError(
                    f"Wrong element type: {self.element.member.element_type}"
                )

            # Auxiliary parameters
            mu = 1 + 12 * Omega
            lamb = 1 + 3 * Omega
            gamma = 1 - 6 * Omega

            # Unitary dimensionless temperature gradient
            tg = (alpha * dty) / h

            # Calculate transversal displacement
            if (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                v = np.zeros(x.shape)

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                v = tg * (
                    (-L * mu / (4 * lamb) + 3 * L * Omega / (2 * lamb)) * x
                    + x**2 / 2
                    - x**3 / (4 * L * lamb)
                )

            elif (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                v = tg * (
                    -3 * L * Omega * x / (2 * lamb)
                    - gamma * x**2 / (4 * lamb)
                    + x**3 / (4 * L * lamb)
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                v = tg * (-L * x / 2 + x**2 / 2)

            else:
                raise AttributeError(
                    f"Wrong hinge type: {self.element.hinge_start}, {self.element.hinge_end}"
                )

        else:
            v = np.zeros(x.shape)

        return v

    def get_flexural_xz_displacements(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Return displacements in local xz-plane at specified positions.

        :param x: An array of positions along the member's local x-axis at which
                  the flexural displacement is to be computed.
        :return: An array of transversal displacements at the `x` positions.
        """
        # Get temperature gradient relative to member local y-axis
        dtz = self.temperature_gradients[2]

        # Check if temperature gradient is not null
        if dtz:
            # Basic member properties
            alpha = self.element.member.section.material.thermal_expansion_coefficient
            h = self.element.member.section.height_z
            L = self.element.length

            # Timoshenko parameter
            if self.element.member.element_type == Element1DType.NAVIER:
                Omega = 0.0

            elif self.element.member.element_type == Element1DType.TIMOSHENKO:
                EI = self.element.member.section.EIy
                GA = self.element.member.section.GAz

                Omega = EI / (GA * L * L)

            else:
                raise AttributeError(
                    f"Wrong element type: {self.element.member.element_type}"
                )

            # Auxiliary parameters
            mu = 1 + 12 * Omega
            lamb = 1 + 3 * Omega
            gamma = 1 - 6 * Omega

            # Unitary dimensionless temperature gradient
            tg = (alpha * dtz) / h

            # Calculate transversal displacement
            if (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                w = np.zeros(x.shape)

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.CONTINUOUS_END
            ):
                w = tg * (
                    (-L * mu / (4 * lamb) + 3 * L * Omega / (2 * lamb)) * x
                    + x**2 / 2
                    - x**3 / (4 * L * lamb)
                )

            elif (
                self.element.hinge_start == BeamConnection.CONTINUOUS_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                w = tg * (
                    -3 * L * Omega * x / (2 * lamb)
                    - gamma * x**2 / (4 * lamb)
                    + x**3 / (4 * L * lamb)
                )

            elif (
                self.element.hinge_start == BeamConnection.HINGED_END
                and self.element.hinge_end == BeamConnection.HINGED_END
            ):
                w = tg * (-L * x / 2 + x**2 / 2)

            else:
                raise AttributeError(
                    f"Wrong hinge type: {self.element.hinge_start}, {self.element.hinge_end}"
                )

        else:
            w = np.zeros(x.shape)

        return w
