from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy as sp  # type: ignore[import-untyped]

from framesss.enums import CoordinateDefinition
from framesss.enums import DistributedLoadLocation
from framesss.enums import LoadCoordinateSystem

if TYPE_CHECKING:
    import numpy.typing as npt

    from framesss.pre.cases import LoadCase
    from framesss.pre.member_1d import Member1D


COORDINATE_DEFINITIONS = [CoordinateDefinition.ABSOLUTE, CoordinateDefinition.RELATIVE]
LOAD_COORDINATE_SYSTEMS = [LoadCoordinateSystem.LOCAL, LoadCoordinateSystem.GLOBAL]


class PointLoadOnMember:
    """
    Represents a point load (forces and moments) acting on a member.

    This class represents a point load applied to a member, allowing specification of the
    load's magnitude and direction, its position along the member, and the coordinate
    system in which the force components are defined.

    :param member: A reference to an instance of the :class:`Member1D` class, representing
                   the member on which the force is acting.
    :param load_components: A list of force components [Fx, Fy, Fz, Mx, My, Mz].
    :param x: The position along the member's local x-axis where the force is applied,
              measured from start node. This value should be within the member's length.
    :param load_case: A reference to an instance of the :class:`LoadCase` class, representing
                      the load case to which this force belongs.
    :param coordinate_system: The coordinate system in which the provided load
                              components are defined. Should be either 'global' or 'local',
                              as defined in :class:`LoadCoordinateSystem`.
    :param coordinate_definition: Specifies the definition of the position. Should be either
                                  'absolute' or 'relative' as defined in :class:`CoordinateDefinition`.
    """

    def __init__(
        self,
        member: Member1D,
        load_components: list[float] | npt.NDArray[np.float64],
        x: float,
        load_case: LoadCase,
        coordinate_system: str | LoadCoordinateSystem,
        coordinate_definition: str | CoordinateDefinition,
    ) -> None:
        """
        Init the PointLoadOnMember class.

        :raises ValueError: If the position 'x' is outside the interval of the member's length.
        """
        self.member = member

        if coordinate_definition == CoordinateDefinition.RELATIVE:
            x *= member.length
        elif coordinate_definition == CoordinateDefinition.ABSOLUTE:
            pass
        else:
            raise ValueError(
                f"Unknown coordinate definition: '{coordinate_definition}.'"
                f"Valid options are: {COORDINATE_DEFINITIONS}."
            )

        if 0 <= x <= member.length:
            self.x = x
        else:
            raise ValueError(
                f"Position of the force is outside the beam interval: [{0, member.length}], x: '{x}'."
            )

        self.components_global, self.components_local = self.set_load_components(
            np.array(load_components), coordinate_system
        )
        self.load_case = load_case

    def __repr__(self) -> str:
        """Return a string representation of PointLoadOnMember class."""
        local_components_str = ", ".join(
            f"{comp:.2f}" for comp in self.components_local
        )
        global_components_str = ", ".join(
            f"{comp:.2f}" for comp in self.components_global
        )
        return (
            f"{self.__class__.__name__}("
            f"member='{self.member.label}', "
            f"local_components=[{local_components_str}], "
            f"global_components=[{global_components_str}], "
            f"x={self.x:.2f}, "
            f"load_case='{self.load_case.label}'"
        )

    def set_load_components(
        self,
        load_components: npt.NDArray[np.float64],
        coordinate_system: str | LoadCoordinateSystem,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Transform the load components based on the specified coordinate system.

        This method takes the components of a point load and transforms them between the
        global coordinate system and local coordinate system of the member.

        :param load_components: Array of the force components [Fx, Fy, Fz, Mx, My, Mz].
        :param coordinate_system: The coordinate system in which the provided load
                                  components are defined. Should be either 'global' or 'local',
                                  as defined in LoadCoordinateSystem. Defaults to 'global'.

        :return: A tuple containing two np.ndarray objects: the first array is the load
                 components in the global coordinate system, and the second is in the local
                 coordinate system.

        :raises ValueError: If the specified coordinate system is neither 'global' nor 'local'.
        """
        load_components = np.array(load_components)

        transformation_matrix = sp.linalg.block_diag(
            self.member.direction_cosine_matrix, self.member.direction_cosine_matrix
        )

        if coordinate_system == LoadCoordinateSystem.GLOBAL:
            components_global = load_components
            components_local = transformation_matrix @ load_components

        elif coordinate_system == LoadCoordinateSystem.LOCAL:
            components_global = transformation_matrix.T @ load_components
            components_local = load_components

        else:
            raise ValueError(
                f"Invalid load coordinate system: '{coordinate_system}'.\n"
                f"Valid options are: {LOAD_COORDINATE_SYSTEMS}"
            )

        return components_global, components_local


class DistributedLoadOnMember:
    """
    Represents a distributed uniform or trapezoidal load acting on a member.

    This class represents a distributed load applied to a member, allowing specification of the
    force's magnitude and direction, its position along the member, and the coordinate
    system in which the force components are defined.

    :param member: A reference to an instance of the :class:`Member1D` class, representing
                   the member on which the force is acting.
    :param load_components: A list of force components [fx1, fy1, fz1, fx2, fy2, fz2].
    :param x_start: The position along the member's local x-axis of the point, where the load
                    begins to act measured from start node.
                    This value should be within the member's length.
    :param x_end: The position along the member's local x-axis of the point, where the load,
                  application ends measured from start node.
                  This value should be within the member's length.
    :param load_case: A reference to an instance of the :class:`LoadCase` class, representing
                      the load case to which this force belongs.
    :param coordinate_system: The coordinate system in which the provided load
                              components are defined. Should be either 'global' or 'local',
                              as defined in :class:`LoadCoordinateSystem`.
    :param coordinate_definition: Specifies the definition of the position. Should be either
                                  'absolute' or 'relative' as defined in :class:`CoordinateDefinition`.
    :param location: Specifies whether the load is acting directly on an inclined 1D member,
                     or on the projection into the global coordinate system.
                     For 'local' coordinate_system, the only option is 'length'.
                     For 'global' coordinate_system, the options are 'length' and 'projection'.
    """

    def __init__(
        self,
        member: Member1D,
        load_components: list[float] | npt.NDArray[np.float64],
        x_start: float,
        x_end: float,
        load_case: LoadCase,
        coordinate_system: str | LoadCoordinateSystem,
        location: str | DistributedLoadLocation,
        coordinate_definition: str | CoordinateDefinition,
    ) -> None:
        """
        Init the DistributedLoadOnMember class.

        :raises ValueError: If the position 'x' is outside the interval of the member's length.
        """
        self.member = member

        if coordinate_definition == CoordinateDefinition.RELATIVE:
            x_start *= member.length
            x_end *= member.length
        elif coordinate_definition == CoordinateDefinition.ABSOLUTE:
            pass
        else:
            raise ValueError(
                f"Unknown coordinate definition: '{coordinate_definition}.'"
                f"Valid options are: {COORDINATE_DEFINITIONS}."
            )

        if x_start > x_end:
            raise ValueError(
                f"Position of the start point is greater than the position of the end point: "
                f"'{x_start}' > '{x_end}'"
            )
        elif x_start == x_end:
            raise ValueError(
                f"Position of the start point is equal to the position of the end point: "
                f"'{x_start}' == '{x_end}'"
            )
        elif not 0 <= x_start <= member.length:
            raise ValueError(
                f"Position of the start point is outside of the beam interval: "
                f"[{0, member.length}], x: '{x_start}'."
            )
        elif not 0 <= x_end <= member.length:
            raise ValueError(
                f"Position of the end point is outside of the beam interval: "
                f"[{0, member.length}], x: '{x_end}'."
            )
        elif 0 <= x_start <= x_end <= member.length:
            self.x_start = x_start
            self.x_end = x_end
        else:
            raise ValueError("wtf just happened.")

        self.components_global, self.components_local = self.set_load_components(
            np.array(load_components), coordinate_system, location
        )
        self.load_case = load_case

    def __repr__(self) -> str:
        """Return string representation of the DistributedLoadOnMember class."""
        local_components_str = ", ".join(
            f"{comp:.2f}" for comp in self.components_local
        )
        global_components_str = ", ".join(
            f"{comp:.2f}" for comp in self.components_global
        )
        return (
            f"{self.__class__.__name__}("
            f"member='{self.member.label}', "
            f"local_components=[{local_components_str}], "
            f"global_components=[{global_components_str}], "
            f"x_start={self.x_start:.2f}, x_end={self.x_end:.2f}, "
            f"load_case='{self.load_case.label}'"
        )

    def set_load_components(
        self,
        load_components: npt.NDArray[np.float64],
        coordinate_system: str | LoadCoordinateSystem,
        location: str | DistributedLoadLocation,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Set the load components of a distributed load based on the specified coordinate system.

        This method takes the components of a distributed load and transforms them between the
        global coordinate system and local coordinate system of the member.

        :param load_components: A list of the force components [fx1, fy1, fz1, fx2, fy2, fz2].
        :param coordinate_system: The coordinate system in which the provided load
                                  components are defined. Should be either 'global' or 'local',
                                  as defined in LoadCoordinateSystem. Defaults to 'global'.
        :param location: Specifies whether the load is acting directly on an inclined 1D member,
                         or on the projection into the global coordinate system.
                         For 'local' coordinate_system, the only option is 'length'.
                         For 'global' coordinate_system, the options are 'length' and 'projection'.


        :return: A tuple containing two np.ndarray objects: the first array is the force
                 components in the global coordinate system, and the second is in the local
                 coordinate system.

        :raises ValueError: If the specified coordinate system is neither 'global' nor 'local'.
        """
        load_components = np.array(load_components)

        transformation_matrix = sp.linalg.block_diag(
            self.member.direction_cosine_matrix, self.member.direction_cosine_matrix
        )

        if (
            location == DistributedLoadLocation.PROJECTION
            and coordinate_system == LoadCoordinateSystem.GLOBAL
        ):
            cosine = np.tile(
                [
                    (1 - self.member.cosine_x**2) ** 0.5,
                    (1 - self.member.cosine_y**2) ** 0.5,
                    (1 - self.member.cosine_z**2) ** 0.5,
                ],
                2,
            )
            load_components = load_components * cosine

        elif (
            location == DistributedLoadLocation.PROJECTION
            and coordinate_system == LoadCoordinateSystem.LOCAL
        ):
            raise AttributeError(
                f"Cannot set projection to: '{DistributedLoadLocation.LENGTH}' and "
                f"coordinate_system to: '{LoadCoordinateSystem.LOCAL}'"
            )

        if coordinate_system == LoadCoordinateSystem.GLOBAL:
            components_global = load_components
            components_local = transformation_matrix @ load_components

        elif coordinate_system == LoadCoordinateSystem.LOCAL:
            components_global = transformation_matrix.T @ load_components
            components_local = load_components

        else:
            raise ValueError(
                f"Invalid load coordinate system: '{coordinate_system}'.\n"
                f"Valid options are: {LOAD_COORDINATE_SYSTEMS}"
            )

        return components_global, components_local

    def get_load_values_at_location(self, x: float) -> npt.NDArray[np.float64]:
        """
        Return the load magnitude at a specific location along the member's x-axis.

        This method interpolates the load values between the start and end points of the
        applied load based on the given position 'x'.

        :param x: The position along the member's length (measured from the start of
                  the member) at which the load values are to be calculated.
        :return: The interpolated load vector at the specified location 'x': [fx, fy, fz].
        """
        f_start = self.components_local[:3]
        f_end = self.components_local[3:]

        return f_start + (x - self.x_start) * (f_end - f_start) / (
            self.x_end - self.x_start
        )


class ThermalLoadOnMember:
    """
    Represents a distributed uniform or trapezoidal load acting on a member.

    This class represents a distributed load applied to a member, allowing specification of the
    force's magnitude and direction, its position along the member, and the coordinate
    system in which the force components are defined.

    :param member: A reference to an instance of the :class:`Member1D` class, representing
                   the member on which the force is acting.
    :param temperature_gradients: A list of the temperature gradients relative to local axis [tgx, tgy, tgz].
    :param x_start: The position along the member's local x-axis of the point, where the load
                    begins to act measured from start node.
                    This value should be within the member's length.
    :param x_end: The position along the member's local x-axis of the point, where the load,
                  application ends measured from start node.
                  This value should be within the member's length.
    :param load_case: A reference to an instance of the :class:`LoadCase` class, representing
                      the load case to which this force belongs.
    :param coordinate_definition: Specifies the definition of the position. Should be either
                                  'absolute' or 'relative' as defined in :class:`CoordinateDefinition`.
    """

    def __init__(
        self,
        member: Member1D,
        temperature_gradients: list[float] | npt.NDArray[np.float64],
        x_start: float,
        x_end: float,
        load_case: LoadCase,
        coordinate_definition: str | CoordinateDefinition,
    ) -> None:
        """
        Init the ThermalLoadOnMember class.

        :raises ValueError: If the position 'x' is outside the interval of the member's length.
        """
        self.member = member

        if coordinate_definition == CoordinateDefinition.RELATIVE:
            x_start *= member.length
            x_end *= member.length
        elif coordinate_definition == CoordinateDefinition.ABSOLUTE:
            pass
        else:
            raise ValueError(
                f"Unknown coordinate definition: '{coordinate_definition}.'"
                f"Valid options are: {COORDINATE_DEFINITIONS}."
            )

        if x_start > x_end:
            raise ValueError(
                f"Position of the start point is greater than the position of the end point: "
                f"'{x_start}' > '{x_end}'"
            )
        elif x_start == x_end:
            raise ValueError(
                f"Position of the start point is equal to the position of the end point: "
                f"'{x_start}' == '{x_end}'"
            )
        elif not 0 <= x_start <= member.length:
            raise ValueError(
                f"Position of the start point is outside of the beam interval: "
                f"[{0, member.length}], x: '{x_start}'."
            )
        elif not 0 <= x_end <= member.length:
            raise ValueError(
                f"Position of the end point is outside of the beam interval: "
                f"[{0, member.length}], x: '{x_end}'."
            )
        elif 0 <= x_start <= x_end <= member.length:
            self.x_start = x_start
            self.x_end = x_end
        else:
            raise ValueError("wtf just happened.")

        self.temperature_gradients = np.array(temperature_gradients)

        self.load_case = load_case
