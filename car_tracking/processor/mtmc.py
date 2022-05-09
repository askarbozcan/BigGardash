from typing import Dict
from ..detector._base import BaseDetector
from ..tracker._base import BaseTracker
from ..nms._base import BaseNMS
from ..mtmc._base import BaseMTMC
from ._base import BaseProcessor

class MTMCProcessor:
    def __init__(self, detector: BaseDetector, trackers: Dict[str, BaseTracker], mtmc: BaseMTMC, nms: BaseNMS = None):
        self.detector = detector
        self.nms = nms
        self.trackers = trackers
        self.mtmc = mtmc

    def update(self, frames_dict):
        """
            frames_dict: Frame at time t from each camera
        """

        boxes_dict, ids_dict, labels_dict = {}, {}, {}

        for cid, frame in frames_dict.items():
            boxes, scores, labels = self.detector.detect_frame(frame)
            if self.nms is not None:
                boxes, scores, labels = self.nms.filter_boxes(frame)
            boxes, ids, labels = self.trackers[cid].update(boxes, scores, labels)

            boxes_dict[cid] = boxes
            ids_dict[cid] = ids
            labels_dict[cid] = labels
            
        matched_ids_list = self.mtmc.update(boxes_dict, ids_dict, labels_dict)

        return boxes_dict, matched_ids_list, labels_dict 



    

    
