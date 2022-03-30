from multiprocessing.process import BaseProcess
from ..detector._base import BaseDetector
from ..tracker._base import BaseTracker
from ..nms._base import BaseNMS
from ..mtmc._base import BaseMTMC
from ._base import BaseProcessor

class MTMCProcessor:
    def __init__(self, detector: BaseDetector, nms: BaseNMS, tracker: BaseTracker, mtmc: BaseMTMC):
        self.detector = detector
        self.nms = nms
        self.tracker = tracker
        self.mtmc = mtmc
    

    
