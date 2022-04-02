from multiprocessing.process import BaseProcess
from ..detector._base import BaseDetector
from ..tracker._base import BaseTracker
from ..nms._base import BaseNMS
from ..mtmc._base import BaseMTMC
from ._base import BaseProcessor

class MTMCProcessor:
    def __init__(self, detector: BaseDetector, tracker: BaseTracker, mtmc: BaseMTMC, nms: BaseNMS = None):
        self.detector = detector
        self.nms = nms
        self.tracker = tracker
        self.mtmc = mtmc

    def update(self, frames_list, camera_ids):
        """
            frames_list: Frame at time t from each camera
        """

        boxes_list, ids_list, labels_list = [], [], []
        for frame in frames_list:
            boxes, scores, labels = self.detector.detect_frame(frame)
            if self.nms is not None:
                boxes, scores, labels = self.nms.filter_boxes(frame)
            boxes, ids, labels = self.tracker.update(boxes, scores, labels)

            boxes_list.append(boxes)
            ids_list.append(ids)
            labels_list.append(labels)
            
        matched_ids_list = self.mtmc.update(boxes_list, ids_list, labels_list, camera_ids)

        return boxes_list, matched_ids_list, labels_list 



    

    
