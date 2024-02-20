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


class PolygonalSection(Section):
    """Class representing a polygonal cross-section."""

    def __init__(self, label: str, points: list[list[float]]) -> None:
        """Init the PolygonalSection class."""
        if points[0] != points[-1]:
            points.append(points[0])
        self.points = points

        self.y = [c[0] for c in self.points]
        self.z = [c[1] for c in self.points]

        self.n_points = len(self.points) - 1

        area = self.area()
        Iy, Iz, Dyz = self.inertia()
        Ix = Iy + Iz
        hy = max(self.y) - min(self.y)
        hz = max(self.z) - min(self.z)
        super().__init__(label, area, area, area, Ix, Iy, Iz, hy, hz)

    def area(self) -> float:
        """Calculate the area of cross-section."""
        y = self.y
        z = self.z
        s = 0
        for i in range(self.n_points):
            s += y[i] * z[i + 1] - y[i + 1] * z[i]
        return s / 2

    def centroid(self) -> tuple[float, float]:
        """Calculate the location of centroid."""
        y = self.y
        z = self.z
        a = self.area()
        sy = sz = 0
        for i in range(self.n_points):
            sy += (y[i] + y[i + 1]) * (y[i] * z[i + 1] - y[i + 1] * z[i])
            sz += (z[i] + z[i + 1]) * (y[i] * z[i + 1] - y[i + 1] * z[i])
        return sy / (6 * a), sz / (6 * a)

    def inertia(self) -> tuple[float, float, float]:
        """Calculate moments and product of inertia about centroid."""
        y = self.y
        z = self.z
        a = self.area()
        cy, cz = self.centroid()
        syy = szz = syz = 0
        for i in range(self.n_points):
            syy += (z[i] ** 2 + z[i] * z[i + 1] + z[i + 1] ** 2) * (
                y[i] * z[i + 1] - y[i + 1] * z[i]
            )
            szz += (y[i] ** 2 + y[i] * y[i + 1] + y[i + 1] ** 2) * (
                y[i] * z[i + 1] - y[i + 1] * z[i]
            )
            syz += (
                y[i] * z[i + 1]
                + 2 * y[i] * z[i]
                + 2 * y[i + 1] * z[i + 1]
                + y[i + 1] * z[i]
            ) * (y[i] * z[i + 1] - y[i + 1] * z[i])
        return syy / 12 - a * cz**2, szz / 12 - a * cy**2, syz / 24 - a * cy * cz


class RectangularSection(PolygonalSection):
    """Class representing a rectangular section."""

    def __init__(self, label: str, b: float, h: float) -> None:
        """Init the RectangularSection class."""
        self.b = b
        self.h = h

        points = [[0.0, 0.0], [b, 0.0], [b, h], [0.0, h]]

        super().__init__(label, points)
