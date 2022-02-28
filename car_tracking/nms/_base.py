from abc import ABC
import numpy as np
from typing import Tuple

class BaseNMS(ABC):
    def filter_boxes(in_boxes: np.ndarray, in_scores: np.ndarray, in_labels: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Given [N*4] in_boxes, [N] box_scores and [N] labels,
            perform NMS. Returns M boxes, M <= N.
                Returns:
                    out boxes [M*4]
                    box scores [M]
                    labels [M]
        """
        pass
