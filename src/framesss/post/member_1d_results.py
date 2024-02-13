from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from framesss.pre.cases import LoadCase
    from framesss.pre.member_1d import Member1D


class Member1DResults:
    """Class for storing results from finite element analysis."""

    def __init__(self, member: Member1D) -> None:
        """Init the Member1DResults class."""
        self.member = member

        # results after analysis is done
        self.translations_x: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.translations_y: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.translations_z: dict[LoadCase, npt.NDArray[np.float64]] = {}

        self.rotations_x: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.rotations_y: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.rotations_z: dict[LoadCase, npt.NDArray[np.float64]] = {}

        self.axial_forces: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.shear_forces_y: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.shear_forces_z: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.torsional_moments: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.bending_moments_y: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.bending_moments_z: dict[LoadCase, npt.NDArray[np.float64]] = {}

        self.extreme_axial_forces: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.extreme_shear_forces_y: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.extreme_shear_forces_z: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.extreme_torsional_moments: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.extreme_bending_moments_y: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.extreme_bending_moments_z: dict[LoadCase, npt.NDArray[np.float64]] = {}

        self.min_max_axial_forces: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.min_max_shear_forces_y: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.min_max_shear_forces_z: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.min_max_torsional_moments: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.min_max_bending_moments_y: dict[LoadCase, npt.NDArray[np.float64]] = {}
        self.min_max_bending_moments_z: dict[LoadCase, npt.NDArray[np.float64]] = {}

    def __repr__(self) -> str:
        """Return a string representation of the Member1DResults object."""
        return f"{self.__class__.__name__}(member={self.member.label})"
