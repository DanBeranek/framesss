from __future__ import annotations

from framesss.pre.load_case import LoadCase


class LoadCombination:
    """
    Represent a combination of load cases.

    :param label: A unique identifier for the load combination.
    :ivar combinations: A dictionary mapping :class:`LoadCase` instances
                        to their scaling factors.
    """

    def __init__(self, label: str, combinations: dict[LoadCase, float]) -> None:
        """Init the LoadCombination class."""
        self.label = label
        self.combinations = combinations

    def __repr__(self) -> str:
        """Return a string representation of LoadCombination object."""
        return f"{self.__class__.__name__}({self.label})"

    def add_load_case(self, load_case: LoadCase, factor: float) -> None:
        """
        Add load case to load combination.

        :param load_case: A reference to an instance of the :class:`LoadCase` class.
        :param factor: The factor for a given load case.
        """
        self.combinations[load_case] = factor
