from abc import ABC
import numpy as np
from typing import Tuple

class BaseTracker(ABC):
    def update(in_boxes: np.ndarray, in_scores: np.ndarray, in_labels: np.ndarray) -> np.ndarray:
        """
            Given [N*4] in_boxes, [N] box_scores and [N] labels,
            perform NMS. Returns M boxes, M <= N.
                Returns:
                    [N*4] boxes, [N] ids and [N] labels

        """
        pass
