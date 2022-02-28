import numpy as np
from numpy.core.fromnumeric import sort
from ._base import BaseTracker
from ._sort_module import Sort as SortTrackerClass

class SORT(BaseTracker):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self._sort = SortTrackerClass(max_age, min_hits, iou_threshold)

    def update(self, in_boxes, in_scores, in_labels):
        #FIXME: Label information is lost for now.
        sort_boxes = np.concatenate([in_boxes, np.reshape(in_scores, (-1,1))], axis=1)
        sort_boxes = self._sort.update(sort_boxes)

        return sort_boxes[:, :4], sort_boxes[:, 4].flatten(), np.ones(sort_boxes.shape[0])*-1

