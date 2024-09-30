from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


class SingularMatrixError(np.linalg.LinAlgError):
    """
    Exception raised when a matrix is singular or nearly singular.

    :param message: The error message to display.
    :param matrix: The matrix that is singular or nearly singular.
    """

    def __init__(
        self,
        message: str = "Singular stiffness matrix.",
        matrix: npt.NDArray | None = None,
    ) -> None:
        """
        Initialize the exception.
        """
        super().__init__(message)
        self.matrix = matrix
