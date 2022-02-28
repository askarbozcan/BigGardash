import cv2
from ._base import BaseProcessor

class VideoProcessor(BaseProcessor):
    def __init__(self, detector, nms=None, tracker=None, skip_N=1):
        super().__init__(detector, nms=nms, tracker=tracker)
        self.cap = None
        self.skip_N = skip_N # process every skip_Nth frame


    def set_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_counter = 0

    def get_timestamp(self):
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)


    def next_frame(self):
        assert self.cap is not None, "Please, set_video() first"
        while self.cap.isOpened():
            ts = self.get_timestamp()
            ret, frame = self.cap.read()
            if self.frame_counter % self.skip_N != 0:
                self.frame_counter += 1
                continue

            if not ret:
                break
            
            self.frame_counter += 1
            yield frame, ts