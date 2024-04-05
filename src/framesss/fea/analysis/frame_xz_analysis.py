from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy as sp  # type: ignore[import-untyped]

from framesss.enums import AnalysisModelType
from framesss.enums import DoF
from framesss.fea.analysis.analysis import Analysis
from framesss.pre.cases import EnvelopeCombination
from framesss.pre.cases import LoadCase
from framesss.utils import assemble_subarray_at_indices

if TYPE_CHECKING:
    import numpy.typing as npt

    from framesss.fea.boundary_conditions.element_load import ElementLoad
    from framesss.fea.element_1d import Element1D
    from framesss.fea.models.model import Model
    from framesss.fea.node import Node
    from framesss.pre.cases import LoadCaseCombination
    from framesss.pre.member_1d import Member1D


class FrameXZAnalysis(Analysis):
    """
    Subclass of the :class:`Analysis` class for the implementation of the 2D Frame model in XZ-plane.

    The :class:`FrameXZAnalysis` model has several key assumptions:
        - Frame elements are typically rigidly connected at joints, but can have a hinge at one or both ends.
        - A hinge in a 2D frame member allows rotation in the out-of-plane direction to be free.
        - The model is assumed to be laid in the XZ-plane, considering only in-plane behavior,
          without displacement transversal to the frame plane.
        - Internal forces at any cross-section of a 2D frame member are

            * axial force (along local x-axis),
            * bending moment (about local y-axis),
            * shear force (along local z-axis).

        - Each node in a 2D frame model has three degrees of freedom (DoFs)

            * horizontal displacement (along global X-axis),
            * rotation (about global Y-axis),
            * vertical displacement (along global Z-axis).

    This subclass specifically implements the behavior and properties unique to a 2D frame model.

    :ivar analysis_type: Specifies the analysis type as FRAME_XZ.
    :ivar active_dofs: Active degrees of freedom for the FrameXZAnalysis.
    :ivar active_rotational_dofs: Active rotational degree of freedom.
    :ivar dof_elem_axial: Indices corresponding to the axial DoFs.
    :ivar dof_elem_flexural_xz: Indices corresponding to the flexural DoFs.
    """

    def __init__(self) -> None:
        """Init the FrameXZAnalysis object."""
        super().__init__(
            analysis_type=AnalysisModelType(AnalysisModelType.FRAME_XZ),
            active_dofs=[
                DoF.TRANSLATION_X,
                DoF.ROTATION_Y,
                DoF.TRANSLATION_Z,
            ],
            active_rotational_dofs=[DoF.ROTATION_Y],
        )
        self.dof_elem_axial = [0, 3]
        self.dof_elem_flexural_xz = [2, 1, 5, 4]

    def get_auxiliary_vector_in_local_xy_plane(
        self, nodes: list[Node]
    ) -> npt.NDArray[np.float64]:
        """
        Calculate and return the auxiliary vector in the local xy-plane for a given list of nodes.

        The auxiliary vector lies in the local xy-plane of a member, and the
        cross-product of the auxiliary vector with the x-axis defines the local
        z-axis vector. In the FrameXZAnalysis context, the auxiliary vector is
        automatically set to [0., 1., 0.].

        :param nodes: A list of :class:`Node` objects representing the nodes of the
                      member for which the auxiliary vector is calculated.
        :return: The auxiliary vector, set to [0., 1., 0.] in the FrameXZAnalysis context.
        """
        return np.array([0.0, 1.0, 0.0])

    def get_transformation_matrix(self, member: Member1D) -> npt.NDArray[np.float64]:
        """
        Compute the transformation matrix for a given structural member.

        :param member: A reference to an instance of the :class:`Member1D` class.
        :return: The 6x6 transformation matrix for the given member.
        """
        direction_cosines = member.direction_cosine_matrix

        transformation_matrix = sp.linalg.block_diag(
            direction_cosines, direction_cosines
        )

        return transformation_matrix  # type: ignore[no-any-return]

    def get_element_local_stiffness_matrix(
        self, element: Element1D
    ) -> npt.NDArray[np.float64]:
        """
        Assembles and returns the local stiffness matrix for a specified element.

        :param element: A reference to an instance of the :class:`Element1D` class.
        :return: The local stiffness matrix of the specified element.
        """
        # Initialize member stiffness matrix in local system
        kel = np.zeros(
            (
                element.number_of_nodes * self.n_dof_per_node,
                element.number_of_nodes * self.n_dof_per_node,
            )
        )

        # Compute axial stiffness coefficients
        axial_coefficients = element.get_axial_stiffness_coefficients()
        assemble_subarray_at_indices(kel, axial_coefficients, self.dof_elem_axial)

        # Compute flexural stiffness coefficients
        flexural_coefficients = element.get_flexural_xz_stiffness_coefficients()
        assemble_subarray_at_indices(
            kel, flexural_coefficients, self.dof_elem_flexural_xz
        )

        return kel

    def assemble_nodal_loads(self, model: Model, load_case: LoadCase) -> None:
        """
        Assemble nodal load components to the global force vector for a given load case.

        This method iterates over all nodal loads defined in a given load case and adds
        their components to the global force vector.

        :param model: A reference to an instance of the :class:`Model` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        for node, load in load_case.nodal_loads.items():
            node_idx = node.id
            # Add applied force in global X coordinate_system
            load_case.f_global[
                model.dof_connectivity_matrix[0, node_idx]
            ] += load.load_components[0]

            # Add applied force in global Z coordinate_system
            load_case.f_global[
                model.dof_connectivity_matrix[2, node_idx]
            ] += load.load_components[2]

            # Add applied moment about global Y coordinate_system
            load_case.f_global[
                model.dof_connectivity_matrix[1, node_idx]
            ] += load.load_components[4]

    def get_fixed_end_forces(self, load: ElementLoad) -> npt.NDArray[np.float64]:
        """
        Return the fixed end force vector.

        :param load: A reference to an instance of the :class:`ElementLoad` class,
                     representing the distributed load applied to the member.
        :return: The 6x1 fixed end force vector in local system for the element under the specified load.
        """
        # Initialize load vector
        fel = np.zeros(load.element.number_of_nodes * self.n_dof_per_node)

        # Compute axial fixed end force components
        fel[self.dof_elem_axial] = load.get_axial_fixed_end_forces()

        # Compute flexural fixed end force components
        fel[self.dof_elem_flexural_xz] = load.get_flexural_xz_fixed_end_forces()

        return fel

    def assemble_internal_forces(
        self, element: Element1D, load_case: LoadCase, fel: npt.NDArray[np.float64]
    ) -> None:
        """
        Assembles contribution of an internal force vector for a given element and load case.

        :param element: A reference to an instance of the :class:`Element1D` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :param fel: The internal force vector in local system for the element.
        """
        element.end_axial_forces[load_case] = fel[[0, 3]]

        element.end_bending_moments_y[load_case] = fel[[1, 4]]

        element.end_shear_forces_z[load_case] = fel[[2, 5]]

    def get_displacement_shape_function_matrix(
        self, element: Element1D
    ) -> npt.NDArray[np.float64]:
        """
        Return displacement shape function matrix evaluated at sampling points of the element.

        :param element: A reference to an instance of the :class:`Element1D` class.
        :return: Displacement shape function matrix.
        """
        # Get number of points
        n_points = element.number_of_sampling_points
        x = element.sampling_points

        # Compute axial displacement shape functions vector
        nu = element.get_axial_displacement_shape_functions(x)

        # Compute transversal displacement shape functions vector
        nv = element.get_flexural_xz_displacement_shape_functions(x)

        # Initialize displacement shape function matrix
        n = np.zeros((2 * n_points, element.number_of_nodes * self.n_dof_per_node))

        # Assemble displacement shape function matrix
        rows, cols = zip(
            *[(i, j) for i in 2 * np.arange(n_points) for j in self.dof_elem_axial]
        )
        n[rows, cols] = nu.T.flatten()

        rows, cols = zip(
            *[
                (i, j)
                for i in 2 * np.arange(n_points) + 1
                for j in self.dof_elem_flexural_xz
            ]
        )
        n[rows, cols] = nv.T.flatten()

        return n

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
        displacements = n @ u_local

        displacements = np.array([displacements[::2], displacements[1::2]])

        return displacements

    def get_internal_displacements_from_local_analysis(
        self, element: Element1D, load_case: LoadCase
    ) -> npt.NDArray[np.float64]:
        """
        Return displacements from loads evaluated at sampling points of the element.

        :param element: A reference to an instance of the :class:`Element1D` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :return: Displacements from loads at the element sampling points.
        """
        # Get number of points
        n_points = element.number_of_sampling_points

        # Initialize displacements vector resulting from local fea
        displacements = np.zeros((2, n_points))

        if distributed_load := load_case.element_distributed_loads.get(element):
            displacements[0, :] += distributed_load.get_axial_displacements(
                element.sampling_points
            )
            displacements[1, :] += distributed_load.get_flexural_xz_displacements(
                element.sampling_points
            )

        if thermla_load := load_case.element_thermal_loads.get(element):
            displacements[0, :] += thermla_load.get_axial_displacements(
                element.sampling_points
            )
            displacements[1, :] += thermla_load.get_flexural_xz_displacements(
                element.sampling_points
            )

        return displacements

    def save_internal_stresses(
        self, member: Member1D, case: LoadCase | LoadCaseCombination
    ) -> None:
        """
        Compute and save internal stresses.

        Computes and saves the internal stresses (axial forces, shear forces, and bending moments)
        for a member under a specified load case. This includes both detailed distributions
        along the member and extreme values for each stress component.

        This method aggregates internal stress data from each :class:`Element1D` of the :class:`Member1D`,
        including axial forces, shear forces in the Z direction, and bending moments about the Y axis.

        :param member: A reference to an instance of the :class:`Member1D` class.
        :param case: A reference to an instance of the :class:`LoadCase` or :class:`LoadCaseCombination` class.
        """
        # Save equation coefficients

        if isinstance(case, LoadCase):
            for element in member.generated_elements:
                element.save_axial_equation_coefficients_for_load_case(case)
                element.save_shear_force_xz_equation_coefficients_for_load_case(case)
                element.save_bending_moment_xz_equation_coefficients_for_load_case(case)
        else:
            for element in member.generated_elements:
                element.save_axial_equation_coefficients_for_load_combination(case)
                element.save_shear_force_xz_equation_coefficients_for_load_combination(
                    case
                )
                element.save_bending_moment_xz_equation_coefficients_for_load_combination(
                    case
                )

        for element in member.generated_elements:
            element.save_peak_points_for_axial_force_eqn(case)
            element.save_peak_points_for_shear_force_xz_eqn(case)
            element.save_peak_points_for_bending_moment_xz_eqn(case)

        # Initialize empty arrays
        axial_forces = np.array([])
        shear_forces_z = np.array([])
        bending_moments_y = np.array([])

        peak_x_local = np.array([])
        peak_axial_forces = np.array([])
        peak_shear_forces_z = np.array([])
        peak_bending_moments_y = np.array([])

        # Loop through every element
        for element in member.generated_elements:
            x = element.sampling_points
            x_peaks = element.peak_points[case]

            peak_x_local = np.append(peak_x_local, x_peaks + element.x_start)

            # Get axial forces
            axial_forces = np.append(axial_forces, element.get_axial_force(case, x))
            peak_axial_forces = np.append(
                peak_axial_forces, element.get_axial_force(case, x_peaks)
            )

            # Get shear forces
            shear_forces_z = np.append(
                shear_forces_z, element.get_shear_force_xz(case, x)
            )
            peak_shear_forces_z = np.append(
                peak_shear_forces_z, element.get_shear_force_xz(case, x_peaks)
            )

            # Get bending moments
            bending_moments_y = np.append(
                bending_moments_y, element.get_bending_moment_xz(case, x)
            )
            peak_bending_moments_y = np.append(
                peak_bending_moments_y, element.get_bending_moment_xz(case, x_peaks)
            )

        # Save results
        # Stack arrays and get only unique records for peak values
        data = np.vstack(
            [
                peak_x_local,
                peak_axial_forces,
                peak_shear_forces_z,
                peak_bending_moments_y,
            ]
        )
        peak_x_local, peak_axial_forces, peak_shear_forces_z, peak_bending_moments_y = (
            np.unique(data, axis=1)
        )

        member.results.peak_x_local[case] = peak_x_local

        member.results.axial_forces[case] = axial_forces
        member.results.peak_axial_forces[case] = peak_axial_forces
        min_max_axial_forces = np.array(
            [np.min(peak_axial_forces), np.max(peak_axial_forces)]
        )
        member.results.min_max_axial_forces[case] = min_max_axial_forces

        member.results.shear_forces_z[case] = shear_forces_z
        member.results.peak_shear_forces_z[case] = peak_shear_forces_z
        min_max_shear_forces_z = np.array(
            [np.min(peak_shear_forces_z), np.max(peak_shear_forces_z)]
        )
        member.results.min_max_shear_forces_z[case] = min_max_shear_forces_z

        member.results.bending_moments_y[case] = bending_moments_y
        member.results.peak_bending_moments_y[case] = peak_bending_moments_y
        min_max_bending_moments_y = np.array(
            [
                np.min(peak_bending_moments_y),
                np.max(peak_bending_moments_y),
            ]
        )
        member.results.min_max_bending_moments_y[case] = min_max_bending_moments_y

    def save_envelope_stresses(
        self, member: Member1D, envelope: EnvelopeCombination
    ) -> None:
        """
        Filter and save the envelope of internal stresses.

        :param member: A reference to an instance of the :class:`Member1D` class.
        :param envelope: A reference to an instance of the :class:`EnvelopeCombination`.
        """
        axial_forces = np.vstack(
            [member.results.axial_forces[case] for case in envelope.cases]
        )
        shear_forces_z = np.vstack(
            [member.results.shear_forces_z[case] for case in envelope.cases]
        )
        bending_moments_y = np.vstack(
            [member.results.bending_moments_y[case] for case in envelope.cases]
        )

        member.results.axial_forces[envelope] = np.array(
            [
                np.min(axial_forces, axis=0),
                np.max(axial_forces, axis=0),
            ]
        )

        member.results.shear_forces_z[envelope] = np.array(
            [
                np.min(shear_forces_z, axis=0),
                np.max(shear_forces_z, axis=0),
            ]
        )

        member.results.bending_moments_y[envelope] = np.array(
            [
                np.min(bending_moments_y, axis=0),
                np.max(bending_moments_y, axis=0),
            ]
        )

    def save_internal_displacements_on_member(
        self, member: Member1D, load_case: LoadCase
    ) -> None:
        """
        Compute and save the internal displacements for a member under a specified load case.

        This method aggregates displacement data from each :class:`Element1D` of the :class:`Member1D`,

        :param member: A reference to an instance of the :class:`Member1D` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        translations_x = np.array([])
        translations_z = np.array([])

        for element in member.generated_elements:
            u_local_from_global_analysis = element.get_element_local_displacements(
                load_case
            )

            u_local_from_local_analysis = (
                self.get_internal_displacements_from_local_analysis(element, load_case)
            )

            u_local = u_local_from_global_analysis + u_local_from_local_analysis

            translations_x = np.append(translations_x, u_local[0, :])
            translations_z = np.append(translations_z, u_local[1, :])

        member.results.translations_x[load_case] = translations_x
        member.results.translations_z[load_case] = translations_z

    def save_internal_displacements_on_member_combination(
        self, member: Member1D, load_combination: LoadCaseCombination
    ) -> None:
        """
        Compute and save the internal displacements for a member under a specified load case.

        This method aggregates displacement data from each :class:`Element1D` of the :class:`Member1D`,

        :param member: A reference to an instance of the :class:`Member1D` class.
        :param load_combination: A reference to an instance of the :class:`LoadCaseCombination` class.
        """
        member.results.translations_x[load_combination] = np.zeros(member.x_local.shape)
        member.results.translations_z[load_combination] = np.zeros(member.x_local.shape)

        for load_case, factor in load_combination.load_cases.items():
            member.results.translations_x[load_combination] += (
                factor * member.results.translations_x[load_case]
            )
            member.results.translations_z[load_combination] += (
                factor * member.results.translations_z[load_case]
            )

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
        if (rfx := node.results.reaction_force_x) is not None:
            rfx[load_case] = load_case.f_global[node.global_dofs[0]]

        if (rmy := node.results.reaction_moment_y) is not None:
            rmy[load_case] = load_case.f_global[node.global_dofs[1]]

        if (rfz := node.results.reaction_force_z) is not None:
            rfz[load_case] = load_case.f_global[node.global_dofs[2]]

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
        for load_case, factor in load_combination.load_cases.items():
            if node.results.reaction_force_x.get(load_case) is not None:
                if node.results.reaction_force_x.get(load_combination) is None:
                    node.results.reaction_force_x[load_combination] = 0
                node.results.reaction_force_x[load_combination] += (
                    factor * load_case.f_global[node.global_dofs[0]]
                )

            if node.results.reaction_moment_y.get(load_case) is not None:
                if node.results.reaction_moment_y.get(load_combination) is None:
                    node.results.reaction_moment_y[load_combination] = 0
                node.results.reaction_moment_y[load_combination] += (
                    factor * load_case.f_global[node.global_dofs[1]]
                )

            if node.results.reaction_force_z.get(load_case) is not None:
                if node.results.reaction_force_z.get(load_combination) is None:
                    node.results.reaction_force_z[load_combination] = 0
                node.results.reaction_force_z[load_combination] += (
                    factor * load_case.f_global[node.global_dofs[2]]
                )

    def save_displacements(self, node: Node, load_case: LoadCase) -> None:
        """
        Save the displacements for a specified node under a given load case.

        This method extracts displacements from the global displacement vector for the
        specified :class:`LoadCase` and assigns them to the corresponding node results.

        :param node: A reference to an instance of the :class:`Node` class.
        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        """
        u_g = load_case.u_global[node.global_dofs]

        node.results.translation_x[load_case] = u_g[0]
        node.results.rotation_y[load_case] = u_g[1]
        node.results.translation_z[load_case] = u_g[2]

    # TODO: docstring
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
        node.results.translation_x[load_combination] = 0.0
        node.results.rotation_y[load_combination] = 0.0
        node.results.translation_z[load_combination] = 0.0

        for load_case, factor in load_combination.load_cases.items():
            u_g = load_case.u_global[node.global_dofs]

            node.results.translation_x[load_combination] += factor * u_g[0]
            node.results.rotation_y[load_combination] += factor * u_g[1]
            node.results.translation_z[load_combination] += factor * u_g[2]
