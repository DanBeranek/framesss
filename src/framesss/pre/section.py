class Section:
    """
    Class for storing the geometric properties of a cross-section.

    :param label: User defined label.
    :param area_x: Area relative to local x-axis (full area).
    :param area_y: Area relative to local y-axis (effective shear_modulus area).
    :param area_z: Area relative to local z-axis (effective shear_modulus area).
    :param inertia_x: Moment of inertia relative to local x-axis (torsion inertia).
    :param inertia_y: Moment of inertia relative to local y-axis (bending inertia).
    :param inertia_z: Moment of inertia relative to local z-axis (bending inertia).
    :param height_y: Height relative to local y-axis.
    :param height_z: Height relative to local z-axis.
    """

    def __init__(
        self,
        label: str,
        area_x: float,
        area_y: float,
        area_z: float,
        inertia_x: float,
        inertia_y: float,
        inertia_z: float,
        height_y: float,
        height_z: float,
    ) -> None:
        """Init the Section class."""
        self.label = label
        self.area_x = area_x
        self.area_y = area_y
        self.area_z = area_z
        self.inertia_x = inertia_x
        self.inertia_y = inertia_y
        self.inertia_z = inertia_z
        self.height_y = height_y
        self.height_z = height_z

    def __repr__(self) -> str:
        """Return a string representation of section."""
        return (
            f"{self.__class__.__name__}("
            f"label='{self.label}', "
            f"area_x={self.area_x:.2e}, area_y={self.area_y:.2e}, area_z={self.area_z:.2e}, "
            f"inertia_x={self.inertia_x:.2e}, inertia_y={self.inertia_y:.2e}, inertia_z={self.inertia_z:.2e}, "
            f"height_y={self.height_y:.2f}, height_z={self.height_z:.2f})"
        )
