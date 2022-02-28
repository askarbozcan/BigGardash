import numpy as np
from collections import deque
from ._base import BaseNMS
from .seqnms_module import seq_nms_scoring_func

class SeqNMS(BaseNMS):
    def __init__(self, frames_N, thresh=0.2, nms_thresh=0.5, linkage_thresh=0.5):
        self.frames_N = frames_N
        self.reset_buffer()

        
        self.thresh = thresh 
        self.nms_thresh = nms_thresh
        self.linkage_thresh = linkage_thresh

    def filter_boxes(self, in_boxes, in_scores, in_labels):

        self.box_buffer.append(in_boxes)
        self.scores_buffer.append(in_scores)
        self.labels_buffer.append(in_labels)
       

        _box_buffer = self.list_to_np_boxes(self.box_buffer)
        _scores_buffer = self.list_to_np(self.scores_buffer, dtype=np.float, pad_val=0.0)
        _labels_buffer = self.list_to_np(self.labels_buffer, dtype=np.int, pad_val=0)

        _boxes = np.stack(_box_buffer, axis=0) # stack along frame dim
        _scores = np.stack(_scores_buffer, axis=0)
        _labels = np.stack(_labels_buffer, axis=0)

        seq_nms_scoring_func(_boxes, _scores, _labels, nms_threshold=self.nms_thresh)

        # return the last frame data
        last_boxes  = _boxes[-1] # boxes are scored in place
        last_scores = _scores[-1]
        last_labels = _labels[-1]

        last_boxes  = last_boxes[last_scores > self.thresh]
        last_labels = last_labels[last_scores > self.thresh]
        last_scores = last_scores[last_scores > self.thresh]

        return last_boxes, last_scores, last_labels

    def reset_buffer(self):
        self.box_buffer = deque([], maxlen=self.frames_N)
        self.scores_buffer = deque([], maxlen=self.frames_N)
        self.labels_buffer = deque([], maxlen=self.frames_N)

    @staticmethod
    def list_to_np(lst, dtype=np.int, pad_val=0):
        pad = max((x.shape[0] for x in lst))
        lst = [np.pad(x, (0, pad-x.shape[0]), constant_values=pad_val) for x in lst]
        return np.array(lst, dtype=dtype)
    
    @staticmethod
    def list_to_np_boxes(lst, dtype=np.int):
        pad_max = max((x.shape[0] for x in lst))
        new_lst = []
        for f_boxes in lst:
            to_pad = pad_max-f_boxes.shape[0]
            if to_pad == 0:
                new_lst.append(f_boxes)
            else:
                pad_arr = np.array([[0,0,0,0] for _ in range(to_pad)], dtype=dtype)
                new_lst.append(np.concatenate((f_boxes, pad_arr), axis=0))

        return np.array(new_lst, dtype=dtype)