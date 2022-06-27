import torch, cv2
import numpy as np
from ._base import BaseDetector
from typing import Tuple

class YOLOV5Detector(BaseDetector):
    def __init__(self, model_str, confidence=0.3, nms_thresh=0.45, label_whitelist=None):
        self.model = torch.hub.load('ultralytics/yolov5', model_str)
        self.model.eval()
        self.model.conf = confidence
        self.model.iou = nms_thresh
        self.label_whitelist = label_whitelist

    def detect_frame(self, frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            dets = self.model(frame).xyxy[0]
        dets = dets.cpu().detach().numpy()

        if self.label_whitelist is not None:
            whitelisted_dets = []
            for l in self.label_whitelist:
                whitelisted_dets.append(dets[dets[:, -1] == float(l)])
            dets = np.concatenate(whitelisted_dets, axis=0)

        boxes = dets[:, :4]
        scores = dets[:, 4]
        labels = dets[:, 5].astype(np.int64)

        return boxes, scores, labels

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', "yolov5s6")
    img = cv2.imread("./screen.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = model(img)
    print(dets.xyxy)