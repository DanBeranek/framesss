import numpy as np


class SingularMatrixError(np.linalg.LinAlgError):
    """
    Exception raised when a matrix is singular or nearly singular.
    """

    def __init__(self, message: str = "Singular stiffness matrix.", matrix=None):
        super().__init__(message)
        self.matrix = matrix
