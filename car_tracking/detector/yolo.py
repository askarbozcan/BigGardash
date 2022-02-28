import sys
import numpy as np
import cv2
from ._base import BaseDetector
from .. import conf
sys.path.append(conf.DARKNET_PATH)
import darknet

class YOLODetector(BaseDetector):
    def __init__(self, config_file, data_file, weights_file, thresh=0.25, allowed_classes=[]):
        self.thresh = thresh

        self.model, self.class_names, self.class_colors = darknet.load_network(
            config_file,
            data_file,
            weights_file,
            batch_size=1
        )

        self.im_w = darknet.network_width(self.model)
        self.im_h = darknet.network_height(self.model)

        # if len(allowed_classes) == 0, assume all classes allowed
        self.allowed_classes = list(set(allowed_classes))
        self._allowed_classes_idx = {v:i for i,v in enumerate(self.allowed_classes)}

    
    def detect_frame(self, frame):
        # frame: frame gotten from cv2 capture
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.im_w, self.im_h),
                                   interpolation=cv2.INTER_LINEAR)
        img_for_detect = darknet.make_image(self.im_w, self.im_h, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

        detections = darknet.detect_image(self.model, self.class_names, img_for_detect, thresh=self.thresh)
        darknet.free_image(img_for_detect)

        # convert to required format
        frame_boxes = []
        frame_scores = []
        frame_labels = []
        for label, confidence, bbox in detections:
            if len(self.allowed_classes) > 0 and label not in self.allowed_classes:
                continue
            frame_boxes.append(self.convert2original(frame, list(bbox)))
            frame_scores.append(float(confidence))
            if len(self.allowed_classes) > 0:
                frame_labels.append(self._allowed_classes_idx[label])
            else:
                frame_labels.append(label)


        boxes, scores, labels = self.get_formatted(frame_boxes, frame_scores, frame_labels)
        
        if len(boxes) == len(scores) == len(labels) == 0:
            boxes = np.zeros((0,4))
            scores = np.zeros((0,))
            labels = np.zeros((0,))
        
        return boxes, scores, labels


    @staticmethod
    def get_formatted(boxes_list, scores_list, labels_list):
        """
            Convert lists of boxes, scores, labels into 
            np arrays
        """

        return np.array(boxes_list, dtype=np.int), \
            np.array(scores_list, dtype=np.float)/100, \
            np.array(labels_list, dtype=np.int),
            # YOLODetector.list_to_np(scores_list, dtype=np.float)/100, \
            # YOLODetector.list_to_np(labels_list, dtype=np.int)


    def convert2relative(self, bbox):
        """
        YOLO format use relative coordinates for annotation
        """
        x, y, w, h  = bbox
        _height     = self.im_h
        _width      = self.im_w
        return x/_width, y/_height, w/_width, h/_height


    def convert2original(self, image, bbox):
        x, y, w, h = self.convert2relative(bbox)

        image_h, image_w, __ = image.shape

        orig_x       = int(x * image_w)
        orig_y       = int(y * image_h)
        orig_width   = int(w * image_w)
        orig_height  = int(h * image_h)

        bbox_converted = (int(orig_x-orig_width/2), int(orig_y-orig_height/2), 
                          int(orig_x + orig_width/2), int(orig_y+orig_height/2))
        return bbox_converted


    def convert4cropping(image, bbox):
        x, y, w, h = self.convert2relative(bbox)

        image_h, image_w, __ = image.shape

        orig_left    = int((x - w / 2.) * image_w)
        orig_right   = int((x + w / 2.) * image_w)
        orig_top     = int((y - h / 2.) * image_h)
        orig_bottom  = int((y + h / 2.) * image_h)

        if (orig_left < 0): orig_left = 0
        if (orig_right > image_w - 1): orig_right = image_w - 1
        if (orig_top < 0): orig_top = 0
        if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

        bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

        return bbox_cropping
