from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from framesss.enums import BeamConnection
from framesss.enums import Element1DType

if TYPE_CHECKING:
    import numpy.typing as npt

    from framesss.fea.node import Node
    from framesss.pre.cases import LoadCase
    from framesss.pre.cases import LoadCaseCombination
    from framesss.pre.member_1d import Member1D

MAX_DISTANCE_BETWEEN_SAMPLING_POINTS = 0.25  # (m)
NUMERIC_GARBAGE = 1.0e-12


class Element1D:
    """
    Class representing a one-dimensional (1D) element in a structural analysis model.

    This class captures the physical and mechanical properties of a 1D element,
    including its connectivity, geometric characteristics, and response to applied loads.

    This class manages a three-dimensional behavior of a linear beam element. An object
    of the :class:`Analysis` class is responsible to project this generic 3D behavior
    to a specific model behavior (2D Truss, 2D Frame, etc.).

    Two different types of flexural behavior are considered:
        - **Euler-Bernoulli (Navier)**: This model assumes the absence of shear deformation.
          Under this assumption, an element subjected to bending maintains its cross-sections
          plane and perpendicular to the longitudinal axis of the element.
        - **Timoshenko**: This model incorporates shear deformation. Under this assumption,
          an element subjected to bending maintains its cross-sections plane but not
          perpendicular to the longitudinal axis.

    To handle different types of flexural behavior, the formulation is based
    on the Timoshenko theory, and it is expressed in a generic way, in which
    the shear parameter ``Omega`` assumes zero value for the Navier theory.

    :ivar member: The parent member this element is part of.
    :ivar nodes: The nodes at the ends of the element.
    :ivar x_start: The local x coordinate of the element's start node along the parent member's length.
    :ivar x_end: The local x coordinate of the element's end node along the parent member's length.
    :ivar hinge_start, hinge_end: Specifies the type of connections at the start and end of the element
                                  (e.g., fixed, hinged, or semirigid) to model the rotational stiffness accurately.
    :ivar id: Unique identification number assigned to the element, initially None until set.
    :ivar number_of_nodes: The number of nodes associated with the element, always 2 for Element1D.
    :ivar length: The length of the element.
    :ivar number_of_sampling_points: The number of points along the element used for detailed analysis.
    :ivar sampling_points: The local x coordinates of the sampling points along the element's length.
    :ivar internal_coords: Coordinates for a set number of points along the element, used for
                           internal analysis and visualization purposes.
    :ivar stiffness_matrix_local: The element's stiffness matrix in the local coordinate system,
                                  initially None until computed.
    :ivar global_dofs: Global degrees of freedom indices for the element, initially None until set.
    :ivar end_axial_forces, end_shear_forces_y, end_shear_forces_z: Dictionaries mapping each LoadCase to the
                            respective internal force at the start of the element.
    :ivar end_torsional_moments, end_bending_moments_y, end_bending_moments_z: Dictionaries mapping each
                            LoadCase to the respective internal moment at the start of the element.
    :ivar peak_points: Dictionary mapping each LoadCase and LoadCaseCombination to the local x coordinate
                       of potential critical points along the element's length.
    """

    def __init__(
        self,
        member: Member1D,
        nodes: list[Node],
        x_start: float,
        x_end: float,
        hinges: list[BeamConnection],
    ) -> None:
        """Init the Element1D class."""
        self.id: None | int = None

        self.number_of_nodes = 2

        self.member = member  # parent member
        self.hinge_start, self.hinge_end = hinges

        self.x_start = x_start
        self.x_end = x_end
        self.length = x_end - x_start

        self.nodes = nodes

        # TODO: Move this to separate method
        # Get nodal coordinates
        xi, yi, zi = nodes[0].coords
        xf, yf, zf = nodes[1].coords

        # Calculate member length
        dx = xf - xi
        dy = yf - yi
        dz = zf - zi
        length = np.sqrt(dx**2 + dy**2 + dz**2)

        # Calculate member cosines with global exes
        cx = dx / length
        cy = dy / length
        cz = dz / length

        # Calculate deformed configuration coordinates of 50 cross-sections
        # along the member local x-axis
        number_of_sampling_points = max(
            int(np.ceil(length / MAX_DISTANCE_BETWEEN_SAMPLING_POINTS)) + 1, 9
        )
        internal_coords = np.ones([3, number_of_sampling_points])
        i = np.linspace(0, length, number_of_sampling_points)
        internal_coords[0, :] = internal_coords[0, :] * xi + i * cx
        internal_coords[1, :] = internal_coords[1, :] * yi + i * cy
        internal_coords[2, :] = internal_coords[2, :] * zi + i * cz

        self.internal_coords = internal_coords
        self.number_of_sampling_points = number_of_sampling_points
        self.sampling_points = np.linspace(0, self.length, number_of_sampling_points)

        self.stiffness_matrix_local: npt.NDArray[np.float64] = np.empty(
            0, dtype=np.float64
        )
        self.global_dofs: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)

        self.end_axial_forces: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.end_shear_forces_y: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.end_shear_forces_z: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.end_torsional_moments: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.end_bending_moments_y: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.end_bending_moments_z: dict[LoadCase, npt.NDArray[np.float64]] = {}

        # [a, b, c]
        self.axial_force_eqn_coefficients: dict[
            LoadCase | LoadCaseCombination, npt.NDArray[np.float64]
        ] = {}
        self.shear_force_y_eqn_coefficients: dict[
            LoadCase | LoadCaseCombination, npt.NDArray[np.float64]
        ] = {}
        self.shear_force_z_eqn_coefficients: dict[
            LoadCase | LoadCaseCombination, npt.NDArray[np.float64]
        ] = {}
        self.torsional_moment_eqn_coefficients: dict[
            LoadCase | LoadCaseCombination, npt.NDArray[np.float64]
        ] = {}

        # [a, b, c, d]
        self.bending_moment_y_eqn_coefficients: dict[
            LoadCase | LoadCaseCombination, npt.NDArray[np.float64]
        ] = {}
        self.bending_moment_z_eqn_coefficients: dict[
            LoadCase | LoadCaseCombination, npt.NDArray[np.float64]
        ] = {}

        self.peak_points: dict[
            LoadCase | LoadCaseCombination, npt.NDArray[np.float64]
        ] = {}

    def __repr__(self) -> str:
        """Return a string representation of the element."""
        node_labels = [node.label for node in self.nodes]
        hinge_descriptions = [hinge for hinge in [self.hinge_start, self.hinge_end]]
        return (
            f"{self.__class__.__name__}("
            f"member={self.member.label}, "
            f"nodes={node_labels}, "
            f"x_start={self.x_start:.2f}, x_end={self.x_end:.2f}, "
            f"length={self.length:.2f}, "
            f"hinges={hinge_descriptions})"
        )

    def get_element_global_stiffness_matrix(self) -> npt.NDArray[np.float64]:
        """
        Return element stiffness matrix in global system.

        :return: Element stiffness matrix in global system.
        """
        # Compute and store member stiffness matrix in local system
        self.stiffness_matrix_local = (
            self.member.analysis.get_element_local_stiffness_matrix(self)
        )

        # Transform member stiffness matrix from local to global system
        keg = (
            self.member.transformation_matrix.T
            @ self.stiffness_matrix_local
            @ self.member.transformation_matrix
        )

        return keg

    def get_element_internal_actions(
        self, load_case: LoadCase
    ) -> npt.NDArray[np.float64]:
        """
        Return element internal actions related to the :class:`LoadCase`.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :return: Internal force vector.
        :raise ValueError: If 'u_global' is None for the case.
        """
        # Get nodal displacements and rotations at member end nodes in global system
        if (u := load_case.u_global) is not None:
            u_g = u[self.global_dofs]
        else:
            raise ValueError("'u_global' is not initialized.")

        # Transform solution from global to local system
        u_l = self.member.transformation_matrix @ u_g

        # Compute internal force vector in local system
        return self.stiffness_matrix_local @ u_l

    def get_axial_stiffness_coefficients(self) -> npt.NDArray[np.float64]:
        """
        Return axial stiffness coefficient matrix.

        :return kea: A 2x2 matrix with axial stiffness coefficients.
        """
        EA = self.member.section.EA
        L = self.length

        k11 = EA / L

        kea = k11 * np.array([[+1, -1], [-1, +1]])

        return kea

    def get_torsion_stiffness_coefficients(self) -> npt.NDArray[np.float64]:
        """
        Return torsion stiffness coefficient matrix.

        :return ket: A 2x2 matrix with torsion stiffness coefficients.
        """
        GJ = self.member.section.GJt
        L = self.length

        if (self.hinge_start == BeamConnection.CONTINUOUS_END) and (
            self.hinge_end == BeamConnection.CONTINUOUS_END
        ):
            k11 = GJ / L

            ket = k11 * np.array([[+1, -1], [-1, +1]])

        else:
            ket = np.zeros([2, 2])

        return ket

    def get_flexural_xy_stiffness_coefficients(self) -> npt.NDArray[np.float64]:
        """
        Return flexural stiffness coefficient matrix in local xy-plane.

        :return kef: A 4x4 matrix with flexural stiffness coefficients.
        """
        EI = self.member.section.EIz
        L = self.length

        if self.member.element_type == Element1DType.NAVIER:
            Omega = 0.0
        elif self.member.element_type == Element1DType.TIMOSHENKO:
            GA = self.member.section.GAy

            Omega = EI / (GA * L * L)
        else:
            raise ValueError(f"Unknown member type: {self.member.element_type}.")

        lamb = 1 + 3 * Omega
        mu = 1 + 12 * Omega
        gamma = 1 - 6 * Omega

        laL = lamb * L
        laL2 = laL * L
        laL3 = laL2 * L

        muL = mu * L
        muL2 = muL * L
        muL3 = muL2 * L

        if (self.hinge_start == BeamConnection.CONTINUOUS_END) and (
            self.hinge_end == BeamConnection.CONTINUOUS_END
        ):
            k11 = 12 * EI / muL3
            k12 = 6 * EI / muL2
            k22 = 4 * lamb * EI / muL
            k24 = 2 * gamma * EI / muL

            kef = np.array(
                [
                    [+k11, +k12, -k11, +k12],
                    [+k12, +k22, -k12, +k24],
                    [-k11, -k12, +k11, -k12],
                    [+k12, +k24, -k12, +k22],
                ]
            )

        elif (self.hinge_start == BeamConnection.HINGED_END) and (
            self.hinge_end == BeamConnection.CONTINUOUS_END
        ):
            k11 = 3 * EI / laL3
            k14 = 3 * EI / laL2
            k44 = 3 * EI / laL

            kef = np.array(
                [
                    [+k11, 0.00, -k11, +k14],
                    [0.00, 0.00, 0.00, 0.00],
                    [-k11, 0.00, +k11, -k14],
                    [+k14, 0.00, -k14, +k44],
                ]
            )

        elif (self.hinge_start == BeamConnection.CONTINUOUS_END) and (
            self.hinge_end == BeamConnection.HINGED_END
        ):
            k11 = 3 * EI / laL3
            k12 = 3 * EI / laL2
            k22 = 3 * EI / laL

            kef = np.array(
                [
                    [+k11, +k12, -k11, 0.00],
                    [+k12, +k22, -k12, 0.00],
                    [-k11, -k12, +k11, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                ]
            )

        elif (self.hinge_start == BeamConnection.HINGED_END) and (
            self.hinge_end == BeamConnection.HINGED_END
        ):
            kef = np.zeros([4, 4])

        else:
            raise ValueError(
                f"Unknown continuity condition at the start or end "
                f"of the member: {self.hinge_start}, {self.hinge_end}."
            )

        return kef

    def get_flexural_xz_stiffness_coefficients(self) -> npt.NDArray[np.float64]:
        """
        Return flexural stiffness coefficient matrix in local xz-plane.

        :return kef: A 4x4 matrix with flexural stiffness coefficients.
        """
        EI = self.member.section.EIy
        L = self.length

        if self.member.element_type == Element1DType.NAVIER:
            Omega = 0.0
        elif self.member.element_type == Element1DType.TIMOSHENKO:
            GA = self.member.section.GAz

            Omega = EI / (GA * L * L)
        else:
            raise ValueError(f"Unknown member type: {self.member.element_type}.")

        lamb = 1 + 3 * Omega
        mu = 1 + 12 * Omega
        gamma = 1 - 6 * Omega

        laL = lamb * L
        laL2 = laL * L
        laL3 = laL2 * L

        muL = mu * L
        muL2 = muL * L
        muL3 = muL2 * L

        if (self.hinge_start == BeamConnection.CONTINUOUS_END) and (
            self.hinge_end == BeamConnection.CONTINUOUS_END
        ):
            k11 = 12 * EI / muL3
            k12 = 6 * EI / muL2
            k22 = 4 * lamb * EI / muL
            k24 = 2 * gamma * EI / muL

            kef = np.array(
                [
                    [+k11, -k12, -k11, -k12],
                    [-k12, +k22, +k12, +k24],
                    [-k11, +k12, +k11, +k12],
                    [-k12, +k24, +k12, +k22],
                ]
            )

        elif (self.hinge_start == BeamConnection.HINGED_END) and (
            self.hinge_end == BeamConnection.CONTINUOUS_END
        ):
            k11 = 3 * EI / laL3
            k14 = 3 * EI / laL2
            k44 = 3 * EI / laL

            kef = np.array(
                [
                    [+k11, 0.00, -k11, -k14],
                    [0.00, 0.00, 0.00, 0.00],
                    [-k11, 0.00, +k11, +k14],
                    [-k14, 0.00, +k14, +k44],
                ]
            )

        elif (self.hinge_start == BeamConnection.CONTINUOUS_END) and (
            self.hinge_end == BeamConnection.HINGED_END
        ):
            k11 = 3 * EI / laL3
            k12 = 3 * EI / laL2
            k22 = 3 * EI / laL

            kef = np.array(
                [
                    [+k11, -k12, -k11, 0.00],
                    [-k12, +k22, +k12, 0.00],
                    [-k11, +k12, +k11, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                ]
            )

        elif (self.hinge_start == BeamConnection.HINGED_END) and (
            self.hinge_end == BeamConnection.HINGED_END
        ):
            kef = np.zeros([4, 4])

        else:
            raise ValueError(
                f"Unknown continuity condition at the start or end "
                f"of the member: {self.hinge_start}, {self.hinge_end}."
            )

        return kef

    def get_axial_displacement_shape_functions(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Return the axial displacement shape functions at a given position(s).

        The axial shape functions are used to interpolate the axial
        displacement(s) of an element.

        :param x: The position(s) along the element's local x-axis.
        :return: A 2x*n* matrix with the evaluated axial displacement
                 shape functions at the specified position(s).
        """
        L = self.length

        return np.array([1 - x / L, x / L])

    def get_flexural_xy_displacement_shape_functions(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Return the flexural displacement shape functions in local xy-plane at a given position(s).

        The flexural shape functions are used to interpolate the
        displacement(s) and rotation(s) of an element.

        :param x: The position(s) along the element's local x-axis.
        :return: A 4x*n* vector with the evaluated flexural displacement
                 shape functions in local xy-plane at the specified position(s).
        """
        n_points = x.size

        L = self.length
        L2 = L * L
        L3 = L2 * L

        if self.member.element_type == Element1DType.NAVIER:
            Omega = 0.0
        elif self.member.element_type == Element1DType.TIMOSHENKO:
            EI = self.member.section.EIz
            GA = self.member.section.GAy

            Omega = EI / (GA * L * L)
        else:
            raise ValueError(f"Unknown member type: {self.member.element_type}.")

        lamb = 1 + 3 * Omega
        mu = 1 + 12 * Omega
        gamma = 1 - 6 * Omega

        if (self.hinge_start == BeamConnection.CONTINUOUS_END) and (
            self.hinge_end == BeamConnection.CONTINUOUS_END
        ):
            Nv1 = (
                1
                - 12 * Omega * x / (L * mu)
                - 3 * x**2 / (L2 * mu)
                + 2 * x**3 / (L3 * mu)
            )
            Nv2 = x - 6 * Omega * x / mu - 2 * lamb * x**2 / (L * mu) + x**3 / (L2 * mu)
            Nv3 = (
                12 * Omega * x / (L * mu) + 3 * x**2 / (L2 * mu) - 2 * x**3 / (L3 * mu)
            )
            Nv4 = -6 * Omega * x / mu - gamma * x**2 / (L * mu) + x**3 / (L2 * mu)

        elif (self.hinge_start == BeamConnection.HINGED_END) and (
            self.hinge_end == BeamConnection.CONTINUOUS_END
        ):
            Nv1 = (
                1
                - 3 * x / (2 * L * lamb)
                - 3 * Omega * x / (L * lamb)
                + x**3 / (2 * L3 * lamb)
            )
            Nv2 = np.zeros(n_points)
            Nv3 = (
                3 * x / (2 * L * lamb)
                + 3 * Omega * x / (L * lamb)
                - x**3 / (2 * L3 * lamb)
            )
            Nv4 = (
                -gamma * x / (2 * lamb) - 3 * Omega * x / lamb + x**3 / (2 * L2 * lamb)
            )

        elif (self.hinge_start == BeamConnection.CONTINUOUS_END) and (
            self.hinge_end == BeamConnection.HINGED_END
        ):
            Nv1 = (
                1
                - 3 * Omega * x / (L * lamb)
                - 3 * x**2 / (2 * L2 * lamb)
                + x**3 / (2 * L3 * lamb)
            )
            Nv2 = (
                x
                - 3 * Omega * x / lamb
                - 3 * x**2 / (2 * L * lamb)
                + x**3 / (2 * L2 * lamb)
            )
            Nv3 = (
                3 * Omega * x / (L * lamb)
                + 3 * x**2 / (2 * L2 * lamb)
                - x**3 / (2 * L3 * lamb)
            )
            Nv4 = np.zeros(n_points)

        elif (self.hinge_start == BeamConnection.HINGED_END) and (
            self.hinge_end == BeamConnection.HINGED_END
        ):
            Nv1 = 1 - x / L
            Nv2 = np.zeros(n_points)
            Nv3 = x / L
            Nv4 = np.zeros(n_points)

        else:
            raise ValueError(
                f"Unknown continuity condition at the start or end "
                f"of the member: {self.hinge_start}, {self.hinge_end}."
            )

        return np.array([Nv1, Nv2, Nv3, Nv4])

    def get_flexural_xz_displacement_shape_functions(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Return the flexural displacement shape functions in local xz-plane at a given position(s).

        The flexural shape functions are used to interpolate the
        displacement(s) and rotation(s) of an element.

        :param x: The position(s) along the element's local x-axis.
        :return: A 4x*n* vector with the evaluated flexural displacement
                 shape functions in local xz-plane at the specified position(s).
        """
        npoints = x.size

        L = self.length
        L2 = L * L
        L3 = L2 * L

        if self.member.element_type == Element1DType.NAVIER:
            Omega = 0.0
        elif self.member.element_type == Element1DType.TIMOSHENKO:
            EI = self.member.section.EIy
            GA = self.member.section.GAz

            Omega = EI / (GA * L * L)
        else:
            raise ValueError(f"Unknown member type: {self.member.element_type}.")

        lamb = 1 + 3 * Omega
        mu = 1 + 12 * Omega
        gamma = 1 - 6 * Omega

        if (self.hinge_start == BeamConnection.CONTINUOUS_END) and (
            self.hinge_end == BeamConnection.CONTINUOUS_END
        ):
            Nw1 = (
                1
                - 12 * Omega * x / (L * mu)
                - 3 * x**2 / (L2 * mu)
                + 2 * x**3 / (L3 * mu)
            )
            Nw2 = (
                -x + 6 * Omega * x / mu + 2 * lamb * x**2 / (L * mu) - x**3 / (L2 * mu)
            )
            Nw3 = (
                12 * Omega * x / (L * mu) + 3 * x**2 / (L2 * mu) - 2 * x**3 / (L3 * mu)
            )
            Nw4 = 6 * Omega * x / mu + gamma * x**2 / (L * mu) - x**3 / (L2 * mu)

        elif (self.hinge_start == BeamConnection.HINGED_END) and (
            self.hinge_end == BeamConnection.CONTINUOUS_END
        ):
            Nw1 = (
                1
                - 3 * x / (2 * L * lamb)
                - 3 * Omega * x / (L * lamb)
                + x**3 / (2 * L3 * lamb)
            )
            Nw2 = np.zeros(npoints)
            Nw3 = (
                3 * x / (2 * L * lamb)
                + 3 * Omega * x / (L * lamb)
                - x**3 / (2 * L3 * lamb)
            )
            Nw4 = gamma * x / (2 * lamb) + 3 * Omega * x / lamb - x**3 / (2 * L2 * lamb)

        elif (self.hinge_start == BeamConnection.CONTINUOUS_END) and (
            self.hinge_end == BeamConnection.HINGED_END
        ):
            Nw1 = (
                1
                - 3 * Omega * x / (L * lamb)
                - 3 * x**2 / (2 * L2 * lamb)
                + x**3 / (2 * L3 * lamb)
            )
            Nw2 = (
                -x
                + 3 * Omega * x / lamb
                + 3 * x**2 / (2 * L * lamb)
                - x**3 / (2 * L2 * lamb)
            )
            Nw3 = (
                3 * Omega * x / (L * lamb)
                + 3 * x**2 / (2 * L2 * lamb)
                - x**3 / (2 * L3 * lamb)
            )
            Nw4 = np.zeros(npoints)

        elif (self.hinge_start == BeamConnection.HINGED_END) and (
            self.hinge_end == BeamConnection.HINGED_END
        ):
            Nw1 = 1 - x / L
            Nw2 = np.zeros(npoints)
            Nw3 = x / L
            Nw4 = np.zeros(npoints)

        else:
            raise ValueError(
                f"Unknown continuity condition at the start or end "
                f"of the member: {self.hinge_start}, {self.hinge_end}."
            )

        return np.array([Nw1, Nw2, Nw3, Nw4])

    def get_element_local_displacements(
        self, load_case: LoadCase
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate member interpolated displacements at sampling points of each element.

        Compute member axial and transversal internal displacement
        vector in local system at a given cross-section position,
        from the global analysis (from nodal displacements and rotations).

        :param load_case: handle to an object of the LoadCase
        :return del: member internal displacements vector at given cross-section positions
        """
        # Get nodal displacements and rotations at member end nodes in global system
        if (u := load_case.u_global) is not None:
            u_global = u[self.global_dofs]
        else:
            raise ValueError("'u_global' is not initialized.")

        # Transform solution from global to local system
        u_local = self.member.transformation_matrix @ u_global

        # Evaluate member shape function matrix at given cross-sections
        N = self.member.analysis.get_displacement_shape_function_matrix(self)

        # Compute internal displacements matrix from global fea
        return self.member.analysis.get_internal_displacements_from_global_analysis(
            N, u_local
        )

    def save_axial_equation_coefficients_for_load_case(
        self, load_case: LoadCase
    ) -> None:
        """
        Calculate axial force coefficients for an element under given load case.

        For linear distributed load, the axial force equation
        is defined as: N(x) = (a / 2) * x^2 + b * x + c.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :return: A tuple of three floats representing the coefficients of the axial force equation.
        """
        n_start = -self.end_axial_forces[load_case][0]

        if abs(n_start) < NUMERIC_GARBAGE:
            n_start = 0.0

        if load := load_case.element_distributed_loads.get(self):
            f_start = +load.components_local[0]
            f_end = +load.components_local[3]
        else:
            # There is no load on element => axial force is constant
            self.axial_force_eqn_coefficients[load_case] = np.array([0.0, 0.0, n_start])
            return

        # Linear load equation coefficients: fx(x) = ax + b
        a, b = get_load_equation_coefficients(f_start, f_end, self.length)

        # Normal force equation coefficients: N(x) = (a / 2) * x^2 + b * x + c
        a, b, c = get_internal_force_coefficients(a, b, n_start)

        self.axial_force_eqn_coefficients[load_case] = np.array([-a, -b, c])

    def save_axial_equation_coefficients_for_load_combination(
        self, load_combination: LoadCaseCombination
    ) -> None:
        """
        Calculate axial force coefficients for an element under given load combination.

        For linear distributed load, the axial force equation
        is defined as: N(x) = (a / 2) * x^2 + b * x + c.

        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        :return: A tuple of three floats representing the coefficients of the axial force equation.
        """
        coefficients = np.zeros(3, dtype=np.float64)

        for load_case, factor in load_combination.load_cases.items():
            coefficients += self.axial_force_eqn_coefficients[load_case] * factor

        self.axial_force_eqn_coefficients[load_combination] = coefficients

    def save_peak_points_for_axial_force_eqn(
        self, case: LoadCase | LoadCaseCombination
    ) -> None:
        """
        Calculate positions of local extreme axial forces for given load case or load combination.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        :return: An array of points where local extremes can occur.
        """
        a, b, c = self.axial_force_eqn_coefficients[case]

        x_peaks = get_suspicious_points([b, 2 * a], self.length)

        if case in self.peak_points:
            peak_points = self.peak_points[case]
            peak_points = np.append(peak_points, x_peaks)
        else:
            peak_points = np.array(x_peaks)

        peak_points = np.unique(peak_points)

        self.peak_points[case] = peak_points

    def get_axial_force(
        self, case: LoadCase | LoadCaseCombination, x: npt.NDArray[np.float64] | float
    ) -> npt.NDArray[np.float64]:
        """
        Calculate axial forces along the element for given load case.

        This method computes the axial force distribution along the element for a specific load
        case. For truss structures, the axial force is considered constant. For other types of
        structures, the axial force is calculated using the coefficients from the axial force equation.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        :param x: An array of positions along the element's x-axis where the internal axial forces are to be computed.
        :return: The calculated axial forces at the specified positions.
        """
        a, b, c = self.axial_force_eqn_coefficients[case]

        return a * x**2 + b * x + c  # type: ignore[no-any-return]

    def save_shear_force_xy_equation_coefficients_for_load_case(
        self, load_case: LoadCase
    ) -> None:
        """
        Calculate shear force coefficients in local xy plane for an element under given load case.

        For linear distributed load, the shear force equation
        is defined as: V(x) = (a / 2) * x^2 + b * x + c.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :return: A tuple of three floats representing the coefficients of the shear force equation.
        """
        shear_start = +self.end_shear_forces_y[load_case][0]

        if abs(shear_start) < NUMERIC_GARBAGE:
            shear_start = 0.0

        if load := load_case.element_distributed_loads.get(self):
            f_start = +load.components_local[1]
            f_end = +load.components_local[4]
        else:
            # There is no load on element => shear force is constant
            self.shear_force_y_eqn_coefficients[load_case] = np.array(
                [0.0, 0.0, shear_start]
            )
            return

        # Linear load equation coefficients: fx(x) = ax + b
        a, b = get_load_equation_coefficients(f_start, f_end, self.length)

        # Normal force equation coefficients: N(x) = (a / 2) * x^2 + b * x + c
        a, b, c = get_internal_force_coefficients(a, b, shear_start)

        self.shear_force_y_eqn_coefficients[load_case] = np.array([a, b, c])

    def save_shear_force_xy_equation_coefficients_for_load_combination(
        self, load_combination: LoadCaseCombination
    ) -> None:
        """
        Calculate shear force coefficients in local xy plane for an element under given load case.

        For linear distributed load, the shear force equation
        is defined as: V(x) = (a / 2) * x^2 + b * x + c.

        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        :return: A tuple of three floats representing the coefficients of the shear force equation.
        """
        coefficients = np.zeros(3, dtype=np.float64)

        for load_case, factor in load_combination.load_cases.items():
            coefficients += self.shear_force_y_eqn_coefficients[load_case] * factor

        self.shear_force_y_eqn_coefficients[load_combination] = coefficients

    def save_peak_points_for_shear_force_xy_eqn(
        self, case: LoadCase | LoadCaseCombination
    ) -> None:
        """
        Calculate positions of local extreme shear forces for given load case or load combination.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        :return: An array of points where local extremes can occur.
        """
        a, b, c = self.shear_force_y_eqn_coefficients[case]

        x_peaks = get_suspicious_points([b, 2 * a], self.length)

        if case in self.peak_points:
            peak_points = self.peak_points[case]
            peak_points = np.append(peak_points, x_peaks)
        else:
            peak_points = np.array(x_peaks)

        peak_points = np.unique(peak_points)

        self.peak_points[case] = peak_points

    def get_shear_force_xy(
        self, case: LoadCase | LoadCaseCombination, x: npt.NDArray[np.float64] | float
    ) -> npt.NDArray[np.float64]:
        """
        Calculate shear forces in local xy plane along the element for given load case.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        :param x: An array of positions along the element's x-axis where the internal axial forces are to be computed.
        :return: The calculated shear forces in local xy plane at the specified positions.
        """
        a, b, c = self.shear_force_y_eqn_coefficients[case]

        return a * x**2 + b * x + c  # type: ignore[no-any-return]

    def save_shear_force_xz_equation_coefficients_for_load_case(
        self, load_case: LoadCase
    ) -> None:
        """
        Calculate shear force coefficients in local xz plane for an element under given load case.

        For linear distributed load, the shear force equation
        is defined as: V(x) = (a / 2) * x^2 + b * x + c.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :return: A tuple of three floats representing the coefficients of the shear force equation.
        """
        shear_start = +self.end_shear_forces_z[load_case][0]

        if abs(shear_start) < NUMERIC_GARBAGE:
            shear_start = 0.0

        if load := load_case.element_distributed_loads.get(self):
            f_start = +load.components_local[2]
            f_end = +load.components_local[5]
        else:
            # There is no load on element => shear force is constant
            self.shear_force_z_eqn_coefficients[load_case] = np.array(
                [0.0, 0.0, shear_start]
            )
            return

        # Linear load equation coefficients: fx(x) = ax + b
        a, b = get_load_equation_coefficients(f_start, f_end, self.length)

        # Normal force equation coefficients: N(x) = (a / 2) * x^2 + b * x + c
        a, b, c = get_internal_force_coefficients(a, b, shear_start)

        self.shear_force_z_eqn_coefficients[load_case] = np.array([a, b, c])

    def save_shear_force_xz_equation_coefficients_for_load_combination(
        self, load_combination: LoadCaseCombination
    ) -> None:
        """
        Calculate shear force coefficients in local xz plane for an element under given load combination.

        For linear distributed load, the shear force equation
        is defined as: V(x) = (a / 2) * x^2 + b * x + c.

        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        :return: A tuple of three floats representing the coefficients of the shear force equation.
        """
        coefficients = np.zeros(3, dtype=np.float64)

        for load_case, factor in load_combination.load_cases.items():
            coefficients += self.shear_force_z_eqn_coefficients[load_case] * factor

        self.shear_force_z_eqn_coefficients[load_combination] = coefficients

    def save_peak_points_for_shear_force_xz_eqn(
        self, case: LoadCase | LoadCaseCombination
    ) -> None:
        """
        Calculate positions of local extreme shear forces for given load case or load combination.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        :return: An array of points where local extremes can occur.
        """
        a, b, c = self.shear_force_z_eqn_coefficients[case]

        x_peaks = get_suspicious_points([b, 2 * a], self.length)

        if case in self.peak_points:
            peak_points = self.peak_points[case]
            peak_points = np.append(peak_points, x_peaks)
        else:
            peak_points = np.array(x_peaks)

        peak_points = np.unique(peak_points)

        self.peak_points[case] = peak_points

    def get_shear_force_xz(
        self, case: LoadCase | LoadCaseCombination, x: npt.NDArray[np.float64] | float
    ) -> npt.NDArray[np.float64]:
        """
        Calculate shear forces in local xz plane along the element for given load case.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        :param x: An array of positions along the element's x-axis where the internal shear forces are to be computed.
        :return: The calculated shear forces in local xz plane at the specified positions.
        """
        a, b, c = self.shear_force_z_eqn_coefficients[case]

        return a * x**2 + b * x + c  # type: ignore[no-any-return]

    def save_bending_moment_xy_equation_coefficients_for_load_case(
        self, load_case: LoadCase
    ) -> None:
        """
        Calculate bending moment coefficients about local z-axis for an element under given load case.

        For linear distributed load, the bending moment equation
        is defined as: M(x) = (a / 6) * x^3 + (b / 2)* x^2 + c * x + d.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :return: A tuple of four floats representing the coefficients of the bending moment equation.
        """
        moment_start = float(-self.end_bending_moments_z[load_case][0])
        shear_start = float(+self.end_shear_forces_y[load_case][0])

        if abs(moment_start) < NUMERIC_GARBAGE:
            moment_start = 0.0
        if abs(shear_start) < NUMERIC_GARBAGE:
            shear_start = 0.0

        if load := load_case.element_distributed_loads.get(self):
            f_start = float(+load.components_local[1])
            f_end = float(+load.components_local[4])
        else:
            # There is no load on element => bending moment is linear
            self.bending_moment_z_eqn_coefficients[load_case] = np.array(
                [0.0, 0.0, shear_start, moment_start]
            )
            return

        # Linear load equation coefficients: fx(x) = ax + b
        a, b = get_load_equation_coefficients(f_start, f_end, self.length)

        # Bending moment equation coefficients: M(x) = (a / 6) * x^3 + (b / 2)* x^2 + c * x + d
        a, b, c, d = get_internal_moment_coefficients(a, b, shear_start, moment_start)

        self.bending_moment_z_eqn_coefficients[load_case] = np.array([a, b, c, d])

    def save_bending_moment_xy_equation_coefficients_for_load_combination(
        self, load_combination: LoadCaseCombination
    ) -> None:
        """
        Calculate bending moment coefficients about local z-axis for an element under given load combination.

        For linear distributed load, the bending moment equation
        is defined as: M(x) = (a / 6) * x^3 + (b / 2)* x^2 + c * x + d.

        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        :return: A tuple of four floats representing the coefficients of the bending moment equation.
        """
        coefficients = np.zeros(4, dtype=np.float64)

        for load_case, factor in load_combination.load_cases.items():
            coefficients += self.bending_moment_z_eqn_coefficients[load_case] * factor

        self.bending_moment_z_eqn_coefficients[load_combination] = coefficients

    def save_peak_points_for_bending_moment_xy_eqn(
        self, case: LoadCase | LoadCaseCombination
    ) -> None:
        """
        Calculate local extreme bending moments about z-axis for given load case.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        :return: An array where each row contains a pair [position, bending moment] for each critical point.
        """
        a, b, c, d = self.bending_moment_z_eqn_coefficients[case]

        x_peaks = get_suspicious_points([c, 2 * b, 3 * a], self.length)

        if case in self.peak_points:
            peak_points = self.peak_points[case]
            peak_points = np.append(peak_points, x_peaks)
        else:
            peak_points = np.array(x_peaks)

        peak_points = np.unique(peak_points)

        self.peak_points[case] = peak_points

    def get_bending_moment_xy(
        self, case: LoadCase | LoadCaseCombination, x: npt.NDArray[np.float64] | float
    ) -> npt.NDArray[np.float64]:
        """
        Calculate bending moment about local z-axis along the element for given load case.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        :param x: An array of positions along the element's x-axis where the internal bending
                  moments are to be computed.
        :return: The calculated bending moments about local z-axis at the specified positions.
        """
        a, b, c, d = self.bending_moment_z_eqn_coefficients[case]

        return a * x**3 + b * x**2 + c * x + d  # type: ignore[no-any-return]

    def save_bending_moment_xz_equation_coefficients_for_load_case(
        self, load_case: LoadCase
    ) -> None:
        """
        Calculate bending moment coefficients about local y-axis for an element under given load case.

        For linear distributed load, the bending moment equation
        is defined as: M(x) = (a / 6) * x^3 + (b / 2)* x^2 + c * x + d.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :return: A tuple of four floats representing the coefficients of the bending moment equation.
        """
        moment_start = float(+self.end_bending_moments_y[load_case][0])
        shear_start = float(+self.end_shear_forces_z[load_case][0])

        if abs(moment_start) < NUMERIC_GARBAGE:
            moment_start = 0.0
        if abs(shear_start) < NUMERIC_GARBAGE:
            shear_start = 0.0

        if load := load_case.element_distributed_loads.get(self):
            f_start = float(+load.components_local[2])
            f_end = float(+load.components_local[5])
        else:
            # There is no load on element => bending moment is linear
            self.bending_moment_y_eqn_coefficients[load_case] = np.array(
                [0.0, 0.0, shear_start, moment_start]
            )
            return

        # Linear load equation coefficients: fx(x) = ax + b
        a, b = get_load_equation_coefficients(f_start, f_end, self.length)

        # Bending moment equation coefficients: M(x) = (a / 6) * x^3 + (b / 2)* x^2 + c * x + d
        a, b, c, d = get_internal_moment_coefficients(a, b, shear_start, moment_start)

        self.bending_moment_y_eqn_coefficients[load_case] = np.array([a, b, c, d])

    def save_bending_moment_xz_equation_coefficients_for_load_combination(
        self, load_combination: LoadCaseCombination
    ) -> None:
        """
        Calculate bending moment coefficients about local y-axis for an element under given load combination.

        For linear distributed load, the bending moment equation
        is defined as: M(x) = (a / 6) * x^3 + (b / 2)* x^2 + c * x + d.

        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        :return: A tuple of four floats representing the coefficients of the bending moment equation.
        """
        coefficients = np.zeros(4, dtype=np.float64)

        for load_case, factor in load_combination.load_cases.items():
            coefficients += self.bending_moment_y_eqn_coefficients[load_case] * factor

        self.bending_moment_y_eqn_coefficients[load_combination] = coefficients

    def save_peak_points_for_bending_moment_xz_eqn(
        self, case: LoadCase | LoadCaseCombination
    ) -> None:
        """
        Calculate local extreme bending moments about y-axis for given load case.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        :return: An array where each row contains a pair [position, bending moment] for each critical point.
        """
        a, b, c, d = self.bending_moment_y_eqn_coefficients[case]

        x_peaks = get_suspicious_points([c, 2 * b, 3 * a], self.length)

        if case in self.peak_points:
            peak_points = self.peak_points[case]
            peak_points = np.append(peak_points, x_peaks)
        else:
            peak_points = np.array(x_peaks)

        peak_points = np.unique(peak_points)

        self.peak_points[case] = peak_points

    def get_bending_moment_xz(
        self, case: LoadCase | LoadCaseCombination, x: npt.NDArray[np.float64] | float
    ) -> npt.NDArray[np.float64]:
        """
        Calculate bending moment about local y-axis along the element for given load case.

        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        :param x: An array of positions along the element's x-axis where the internal bending moments
                  are to be computed.
        :return: The calculated bending moments about local y-axis at the specified positions.
        """
        a, b, c, d = self.bending_moment_y_eqn_coefficients[case]

        return a * x**3 + b * x**2 + c * x + d  # type: ignore[no-any-return]


def get_load_equation_coefficients(
    f_start: float, f_end: float, length: float
) -> tuple[float, float]:
    """
    Calculate the coefficients of the linear load equation 'f(x) = a * x + b'.

    Given the start and end load values over a specified length, this function computes
    the coefficients for the linear equation representing the load variation along that
    length. The load is assumed to vary linearly between the start and end values. The
    equation for the load at any point is given by 'f(x) = a * x + b', where 'a' and 'b'
    are the coefficients computed by this function.

    :param f_start: The load value at the start of the element.
    :param f_end: The load value at the end of the element.
    :param length: The length over which the load varies.
    :return: A tuple containing the coefficients 'a' and 'b' of the linear load equation.
    """
    a = (f_end - f_start) / length
    b = f_start

    return a, b


def get_internal_force_coefficients(
    a: float, b: float, int_force_start: float
) -> tuple[float, float, float]:
    """
    Calculate the coefficients of the axial/shear force equation.

    The equation is in form 'N(x) = (a / 2) * x^2 + b * x + c'.

    Given the coefficients 'a' and 'b' from the linear load equation and the initial internal axial
    or shear force at the start of the element, this function computes the coefficients for the
    internal force equation. The internal force equation is used to determine the axial or shear force
    distribution along a member subjected to a linearly distributed load.

    :param a: The 'a' coefficient from the linear load equation, related to the slope of the load.
    :param b: The 'b' coefficient from the linear load equation, related to the load intercept.
    :param int_force_start: The initial internal force (axial or shear) at the start of the element.
    :return: A tuple containing the coefficients 'a', 'b' and 'c' for the internal force equation.
    """
    a = a / 2
    b = b
    c = int_force_start

    return a, b, c


def get_internal_moment_coefficients(
    a: float, b: float, shear_start: float, moment_start: float
) -> tuple[float, float, float, float]:
    """
    Calculate the coefficients for the bending moment equation.

     The equation is in form 'M(x) = a * x^3 / 6 + b * x^2 / 2 + c * x + d'.

    This function computes the coefficients of the bending moment equation for an element
    subjected to linearly varying load. Given the load coefficients 'a' and 'b' and the
    initial shear force and bending moment at the start of the element, it Calculate the
    corresponding moment equation coefficients.

    :param a: The 'a' coefficient from the linear load equation, related to the slope of the load.
    :param b: The 'b' coefficient from the linear load equation, related to the load intercept.
    :param shear_start: The initial shear force at the start of the element.
    :param moment_start: The initial bending moment at the start of the element.
    :return: A tuple containing the coefficients 'a', 'b', 'c', and 'd' for the bending moment equation.
    """
    a = a / 6
    b = b / 2
    c = shear_start
    d = moment_start

    return a, b, c, d


def get_suspicious_points(
    coefficients: list[float], length: float
) -> npt.NDArray[np.float64]:
    """
    Calculate the real roots of a polynomial within a specified range and return them.

    Given the coefficients of a polynomial and a length of the element, this function
    finds the roots of the polynomial and filters out those that are real and fall within
    the range of 0 to element's length. This is useful for determining critical points
    (like maxima or minima) within a specific element.

    :param coefficients: Coefficients of the polynomial for which the roots are to be calculated.
    :param length: The upper bound of the range within which the real roots are considered valid.
    :return: An array of real roots that lie within the specified range [0, length].
    """
    x_max = np.polynomial.polynomial.polyroots(coefficients)  # type: ignore[no-untyped-call]
    if x_max.size > 0:
        x_max = x_max.real[np.isreal(x_max) * (0 <= x_max) * (x_max <= length)]
    x_max = np.insert(x_max, 0, 0.0)
    x_max = np.append(x_max, length)

    return np.unique(x_max)  # type: ignore[no-any-return]
