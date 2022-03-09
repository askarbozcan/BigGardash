import torch, cv2
import numpy as np
from ._base import BaseDetector
from typing import Tuple

class YOLOV5Detector(BaseDetector):
    def __init__(self, model_str, confidence=0.3, nms_thresh=0.45):
        self.model = torch.hub.load('ultralytics/yolov5', model_str)
        self.model.eval()
        self.model.conf = confidence
        self.model.iou = nms_thresh

    def detect_frame(self, frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dets = self.model(frame).xyxy
        boxes = dets[0][:, :4].cpu().detach().numpy()
        scores = dets[0][:, 4].cpu().detach().numpy()
        labels = dets[0][:, 5].cpu().detach().numpy().astype(np.int64)
        return boxes, scores, labels

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', "yolov5s6")
    img = cv2.imread("./screen.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = model(img)
    print(dets.xyxy)