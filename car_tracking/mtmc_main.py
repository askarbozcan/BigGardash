import numpy as np
import os
from os.path import join as pjoin
import glob
import cv2
import sys
from . import conf

import click

from .detector import YOLOV5Detector
from .nms import SeqNMS
from .tracker import SORT
from .artist import TrackingArtist
from .mtmc import DummyMTMC
from .datasets import AICityDataset

@click.command()
@click.option("-d", "--dataset_path", help="Dataset path", default=None, show_default=True)
@click.option("-i", "--scenario_id", help="Scenario id", default=None, show_default=True)
@click.option("-s", "--split_name", help="Split name", default="train", show_default=True)
@click.option("-m", "--model_str", help="Pretrained model string", default="yolov5m6", show_default=True)
def main(dataset_path, scenario_id, split_name, model_str):
    ds = AICityDataset(dataset_path, split_name).scenarios[scenario_id]


    mtmc = DummyMTMC()
    detector = YOLOV5Detector(model_str, confidence=.2, label_whitelist=[2,3,5,7])
    artist = TrackingArtist()
    # nms = SeqNMS(conf.SEQ_NMS_N, conf.SEQ_NMS_THRESH, conf.SEQ_NMS_NMS_THRESH)

    caps = {}
    trackers = {}
    for cid, cam in ds.cameras.items():
        caps[cid] = cv2.VideoCapture(cam.video_path)
        trackers[cid] = SORT()

    while True:
        for cid, cap in caps.items():
            frame_counter = 0

            if caps[cid].isOpened():
                ret, frame = caps[cid].read()
                if frame_counter % conf.SKIP_N != 0:
                    frame_counter += 1
                    continue

                if not ret:
                    break

                boxes, scores, labels = detector.detect_frame(frame)
                # boxes, ids, labels = nms.filter_boxes(boxes, scores, labels)
                boxes, ids, labels = trackers[cid].update(boxes, scores, labels)
                frame = artist.draw(frame, boxes, ids, labels)

                cv2.imshow(cid, frame)
                cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()