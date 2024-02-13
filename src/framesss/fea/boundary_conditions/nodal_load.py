from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class NodalLoad:
    """
    Class representing load acting directly on :class:`Node`.

    :ivar load_components: The vector of applied loads [Fx, Fy, Fz, Mx, My, Mz].
    """

    def __init__(self) -> None:
        """Init the NodalLoad class."""
        self.load_components: npt.NDArray[np.float64] = np.zeros(6, dtype=np.float64)

    def __repr__(self) -> str:
        """Return string representation of the NodalLoad class."""
        components_str = ", ".join(f"{comp:.2f}" for comp in self.load_components)
        return f"{self.__class__.__name__}(load_components=[{components_str}])"
