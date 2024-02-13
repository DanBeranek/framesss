class Material:
    """
    Class for structural materials.

    :param label: User defined label.
    :param elastic_modulus: The modulus of elasticity of the material.
    :param poissons_ratio: Poisson's ratio of the material.
    :param thermal_expansion_coefficient: The thermal expansion coefficient.
    :param density: The density of the material.

    :ivar shear_modulus: Shear modulus of the material.

    The shear modulus is derived from the elastic modulus and Poisson's ratio using the formula:
    G = E / [2 * (1 + v)], where G is the shear modulus, E is the elastic modulus, and v is Poisson's ratio.
    """

    def __init__(
        self,
        label: str,
        elastic_modulus: float,
        poissons_ratio: float,
        thermal_expansion_coefficient: float,
        density: float,
    ) -> None:
        """Init the Material object."""
        self.label = label
        self.elastic_modulus = elastic_modulus
        self.poissons_ratio = poissons_ratio
        self.thermal_expansion_coefficient = thermal_expansion_coefficient
        self.density = density

        self.shear_modulus = elastic_modulus / (2 * (1 + poissons_ratio))

    def __repr__(self) -> str:
        """Return a string representation of the Material object."""
        return (
            f"{self.__class__.__name__}("
            f"label='{self.label}', "
            f"elastic_modulus={self.elastic_modulus:.2e}, "
            f"poissons_ratio={self.poissons_ratio:.2f}, "
            f"shear_modulus={self.shear_modulus:.2e}, "
            f"thermal_expansion_coefficient={self.thermal_expansion_coefficient:.2e}, "
            f"density={self.density:.2f})"
        )
