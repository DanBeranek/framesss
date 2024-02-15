from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Union

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from framesss.pre.cases import LoadCase
    from framesss.pre.cases import LoadCombination
    from framesss.pre.member_1d import Member1D

    LoadType: TypeAlias = Union[LoadCase, LoadCombination]
    ResultDict: TypeAlias = dict[LoadType, npt.NDArray[np.float64]]


class Member1DResults:
    """Class for storing results from finite element analysis."""

    def __init__(self, member: Member1D) -> None:
        """Init the Member1DResults class."""
        self.member = member

        # results after analysis is done
        self.translations_x: ResultDict = {}
        self.translations_y: ResultDict = {}
        self.translations_z: ResultDict = {}

        self.rotations_x: ResultDict = {}
        self.rotations_y: ResultDict = {}
        self.rotations_z: ResultDict = {}

        self.axial_forces: ResultDict = {}
        self.shear_forces_y: ResultDict = {}
        self.shear_forces_z: ResultDict = {}
        self.torsional_moments: ResultDict = {}
        self.bending_moments_y: ResultDict = {}
        self.bending_moments_z: ResultDict = {}

        self.extreme_axial_forces: ResultDict = {}
        self.extreme_shear_forces_y: ResultDict = {}
        self.extreme_shear_forces_z: ResultDict = {}
        self.extreme_torsional_moments: ResultDict = {}
        self.extreme_bending_moments_y: ResultDict = {}
        self.extreme_bending_moments_z: ResultDict = {}

        self.min_max_axial_forces: ResultDict = {}
        self.min_max_shear_forces_y: ResultDict = {}
        self.min_max_shear_forces_z: ResultDict = {}
        self.min_max_torsional_moments: ResultDict = {}
        self.min_max_bending_moments_y: ResultDict = {}
        self.min_max_bending_moments_z: ResultDict = {}

    def __repr__(self) -> str:
        """Return a string representation of the Member1DResults object."""
        return f"{self.__class__.__name__}(member={self.member.label})"
