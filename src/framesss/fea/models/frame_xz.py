from framesss.fea.analysis.frame_xz_analysis import FrameXZAnalysis
from framesss.fea.models.model import Model


class FrameXZModel(Model):
    """Subclass of the :class:`Model` class for the implementation of the 2D Frame model in XZ-plane."""

    def __init__(self) -> None:
        """Init the FrameXZModel object."""
        super().__init__(analysis=FrameXZAnalysis())
