from enum import Enum
from typing import Any

from aenum import StrEnum  # type: ignore


class CaseInsensitiveStrEnum(StrEnum):  # type: ignore
    """An enumeration that extends :class:`StrEnum` to support case-insensitive string comparisons."""

    @classmethod
    def _missing_(cls, value: str) -> Any:
        """
        Handle case-insensitive value lookup for the enumeration.

        This method is invoked by the base Enum class when a value is not immediately
        found in the enum members. It normalizes the input `value` to lowercase and
        attempts to match it against the lowercase versions of the enum's values. If a match
        is found, the corresponding enum member is returned, allowing for case-insensitive
        matching. If no match is found, a ValueError is raised with a message indicating
        the valid choices.

        :param value: The string value to find in the enum.
        :return: The enum member corresponding to the input `value`, matched case-insensitively.
        :raises ValueError: If the normalized `value` does not match any member of the enum,
                            indicating the value is not a valid choice. The error message
                            includes the valid choices for convenience.
        """
        value = value.lower()
        choices = [member.value for member in cls]  # type: ignore[attr-defined]

        if value not in choices:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}, choose from {choices}."
            )
        for member in cls:  # type: ignore[attr-defined]
            if member.value == value:
                return member


class AnalysisModelType(CaseInsensitiveStrEnum):
    """
    Enumeration of finite element analysis (FEA) model types.

    :cvar TRUSS_XZ: Represents a truss model in the XZ plane. Truss models are composed of elements
                    that can only carry axial loads, making them suitable for analyzing structures
                    that primarily experience tension and compression.
    :cvar FRAME_XZ: Represents a frame model in the XZ plane. Frame models include elements that can
                    carry axial loads, shear forces, and bending moments, allowing for the analysis
                    of structures with rigid connections.
    :cvar GRID_XY: Represents a grid model in the XY plane. Grid models are used for analyzing structures
                   that consist of horizontal beams in a flat plane, often used for floors.

    :cvar TRUSS_XYZ: Represents a three-dimensional truss model. Like its two-dimensional counterpart,
                    it can only carry axial loads but in three dimensions, allowing for the analysis
                    of complex spatial trusses.

    :cvar FRAME_XYZ: Represents a three-dimensional frame model. This model type extends the capabilities
                     of the FRAME_XZ model into three dimensions, suitable for comprehensive analysis
                     of buildings and other structures with rigid connections in a three-dimensional space.
    """

    TRUSS_XZ = "truss_xz"
    FRAME_XZ = "frame_xz"
    GRID_XY = "grid_xy"
    TRUSS_XYZ = "truss_xyz"
    FRAME_XYZ = "frame_xyz"


class Element1DType(CaseInsensitiveStrEnum):
    """
    Enumeration of types of 1D elements based on beam theory.

    :cvar NAVIER: Represents elements based on Euler-Bernoulli beam theory, which assumes that plane sections
                  remain plane and perpendicular to the neutral axis of the beam after deformation. This theory
                  is suitable for slender beams where the shear deformation is negligible.
    :cvar TIMOSHENKO: Represents elements based on Timoshenko beam theory, which accounts for both bending and
                      shear deformations. This theory provides more accurate results for short, thick beams where
                      shear deformation cannot be ignored.
    """

    NAVIER = "navier"
    TIMOSHENKO = "timoshenko"


class BeamConnection(CaseInsensitiveStrEnum):
    """
    Enumeration of types of end conditions for beam elements defining how beams are connected to nodes.

    :cvar HINGED_END: Represents a hinged connection that allows rotation but no translation, effectively
                      releasing moment transfer at the end of the beam.
    :cvar CONTINUOUS_END: Represents a fixed (rigid) connection that prevents both rotation and translation,
                          allowing moment transfer and ensuring continuity of displacements across the connection.
    :cvar SEMIRIGID_END: Describes a connection that partially restrains rotation, providing an intermediate condition
                         between fully hinged and fully fixed. The degree of rigidity can vary and requires further
                         specification of rotational stiffness.
    """

    HINGED_END = "hinged"
    CONTINUOUS_END = "fixed"
    SEMIRIGID_END = "semirigid"


class DistributedLoadDirection(CaseInsensitiveStrEnum):
    """
    Enumeration of possible directions for a distributed load in structural models.

    This enumeration defines the standard axes along which distributed loads can be applied to elements.

    :cvar X: Represents a distributed load applied along the global X-axis or local x-axis.
    :cvar Y: Represents a distributed load applied along the global Y-axis or local y-axis.
    :cvar Z: Represents a distributed load applied along the global Z-axis or local z-axis.
    """

    X = "x"
    Y = "y"
    Z = "z"


class LoadCoordinateSystem(CaseInsensitiveStrEnum):
    """
    Enumeration of coordinate systems used to define the orientation of distributed loads in structural model.

    The choice of coordinate system affects how the direction and magnitude of distributed loads are interpreted.

    :cvar GLOBAL: Specifies that the distributed load is defined in the global coordinate system.
    :cvar LOCAL: Specifies that the distributed load is defined in the local coordinate system of a beam.
    """

    GLOBAL = "global"
    LOCAL = "local"


class DistributedLoadLocation(CaseInsensitiveStrEnum):
    """
    Enumeration for specifying the method of applying a distributed load on an inclined beam.

    :cvar LENGTH: The load is put directly on an inclined 1D member (acts along the entire beam length).
    :cvar PROJECTION: The load acts in projection of the beam to the specified plane, which is useful
                      for modelling the effects of snow loads.
    """

    LENGTH = "length"
    PROJECTION = "projection"


class CoordinateDefinition(CaseInsensitiveStrEnum):
    """
    Enumeration distinguishing between absolute and relative coordinates.

    Enum is necessary for specifying the position of loads, nodes, and other features along a beam.

    :cvar ABSOLUTE: Specifies that a position along the beam is defined in fixed units from the start of the beam.
    :cvar RELATIVE: Specifies that a position along the beam is defined in relative to the total length of the beam.
    """

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class SupportFixity(CaseInsensitiveStrEnum):
    """
    Enumeration of support types.

    :cvar FICTFIXED_DOF: Represents a fictitiously fixed DoF at nodes where all connected
                         elements are hinged to force structure global stability during
                         analysis by avoiding singular stiffness matrix
    :cvar FREE_DOF: Represents a free DoF, implying no constraint is applied to this DoF.
    :cvar FIXED_DOF: Represents a fully constrained DoF, only prescribed displacement or
                     rotation can be applied to this DoF.
    :cvar SPRING_DOF: Represents a spring support, introducing a compliance in the boundary
                      condition that must be further defined by a spring stiffness value.
    """

    FICTFIXED_DOF = "fictfixed"
    FREE_DOF = "free"
    FIXED_DOF = "fixed"
    SPRING_DOF = "spring"


class DoF(Enum):
    """
    Enumeration for all degree of freedoms (DoF).

    This enum classifies different types of movements or rotations in a three-dimensional space.
    Each degree of freedom corresponds to a specific translation along or rotation about one
    of the three principal axes (X, Y, Z).

    :cvar TRANSLATION_X: Translation along the global X-axis.
    :cvar TRANSLATION_Y: Translation along the global Y-axis.
    :cvar TRANSLATION_Z: Translation along the global Z-axis.
    :cvar ROTATION_X: Rotation about the global X-axis.
    :cvar ROTATION_Y: Rotation about the global Y-axis.
    :cvar ROTATION_Z: Rotation about the global Z-axis.
    """

    TRANSLATION_X: tuple[int, str] = (0, "x")
    TRANSLATION_Y: tuple[int, str] = (1, "y")
    TRANSLATION_Z: tuple[int, str] = (2, "z")
    ROTATION_X: tuple[int, str] = (3, "rx")
    ROTATION_Y: tuple[int, str] = (4, "ry")
    ROTATION_Z: tuple[int, str] = (5, "rz")

    def __init__(self, index: int, direction: str) -> None:
        """
        Initialize a DoF instance with an index and direction identifier.

        :param index: The numerical index representing the index of the degree of freedom.
        :param direction: A string identifier representing the direction of the degree of freedom.
        """
        self.index = index
        self.direction = direction

    @staticmethod
    def get_index(direction: str) -> int:
        """
        Retrieve the index corresponding to a given direction of degree of freedom.

        :param direction: The direction identifier of the degree of freedom.
        :return: The index associated with the specified direction.
        :raises ValueError: If no matching degree of freedom is found for the given direction.
        """
        for dof in DoF:
            if dof.direction == direction:
                return dof.index
        raise ValueError(f"No matching DoF found for direction: '{direction}'.")
