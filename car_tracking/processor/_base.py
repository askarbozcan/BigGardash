from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class BaseProcessor(ABC):
    def __init__(self, detector, nms=None, tracker=None):
        self.detector = detector
        self.nms = nms
        self.tracker = tracker
        
    @abstractmethod
    def next_frame(self):
        """
            Iterator that should yield a frame and timestamp associated with it.
        """
        pass

    def process(self):
        """
            Iterator that processes frames depending on
            which modules were added to Processor
            Either yields (if no tracker present):
                timestamp, boxes, scores, labels: Float, [N*4], [N], [N]
            or:
               timestamp, boxes, ids, labels:  Float, [N*4], [N], [N]
        """

        for frame, ts in self.next_frame():
            boxes, scores, labels = self.detector.detect_frame(frame)

            if self.nms is not None:
                boxes, scores, labels = self.nms.filter_boxes(boxes, scores, labels)
            
            if self.tracker is not None:
                boxes, ids, labels = self.tracker.update(boxes, scores, labels)
                yield ts, boxes, ids, labels
            else:
                yield ts, boxes, scores, labels
            