from abc import ABC
import numpy as np
from typing import Tuple

class BaseDetector(ABC):
    """
        Abstract base class for all detectors.
    """

    def detect_frame(self, frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Given a frame (input type depends on specific detector),
            Return:
                bounding box [N*4] (np.ndarray)
                box scores [N] (np.ndarray)
                labels [N] (np.ndarray)
        """
        pass