from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class PrescribedDisplacement:
    """
    Class representing prescribed displacement at the :class:`Node`.

    :ivar prescribed_displacements: The vector of prescribed translations and rotations [ux, uy, uz, rx, ry, rz].

    **Note:** During the assembly of the global system of equations, only prescribed displacements for
    constrained DoFs are considered, making the correct application of these conditions essential
    for structural analysis.
    """

    def __init__(self) -> None:
        """Init the PrescribedDisplacement class."""
        # TODO: This looks like there is a prescribed displacement for each DoF, which is not true.
        #       However, when localizing to a displacement vector, only prescribed displacements
        #       at fixed nodes are considered, so it doesn't matter that zero prescribed displacement
        #       is defined here even for free DoFs.
        self.prescribed_displacements: npt.NDArray[np.float64] = np.zeros(
            6, dtype=np.float64
        )
