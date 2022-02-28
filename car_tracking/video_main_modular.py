import numpy as np
import os
from os.path import join as pjoin
import glob
import cv2
import sys
import motmetrics as mm
from . import conf

import click

sys.path.append(conf.DARKNET_PATH)

from .detector import YOLODetector
from .nms import SeqNMS
from .tracker import SORT
from .artist import TrackingArtist


def _get_yolo_cfgs(model_path):
    assert os.path.isdir(model_path), f"{model_path} is not a proper directory"

    cfg_file     = glob.glob(pjoin(model_path, "*.cfg"))
    data_file    = glob.glob(pjoin(model_path, "*.data"))
    weights_file = glob.glob(pjoin(model_path, "*.weights"))

    assert len(cfg_file) == len(data_file) == len(weights_file) == 1, \
           "Wrong/missing directory/file paths? Make sure model folder contains \
            one .cfg, one .data, one .weights and one .names file"
    
    return cfg_file[0], data_file[0], weights_file[0]


@click.command()
@click.option("-v","--video_input", help="Video file input")
@click.option("-r", "--record", help="Record path", default=None, show_default=True)
@click.option("-f", "--model_folder", help="Pretrained model folder path (containing .weights, .cfg, etc)", default="./pretrained_models/yolov4-tiny", show_default=True)

def main(video_input, record, model_folder):
    delay_done = not conf.ENABLE_DELAY_INIT
    
    input_path = video_input
    cap = cv2.VideoCapture(input_path)

    cfg_file, data_file, weights_file = _get_yolo_cfgs(model_folder)

    detector = YOLODetector(cfg_file, 
                            data_file, 
                            weights_file,
                            thresh=conf.YOLO_THRESH,
                            allowed_classes=conf.ALLOWED_CLASSES)

    nms = SeqNMS(conf.SEQ_NMS_N, conf.SEQ_NMS_THRESH, conf.SEQ_NMS_NMS_THRESH)
    mot_tracker = SORT()
    artist = TrackingArtist()

    frame_counter = 0
    ret, frame = cap.read()
    if record is not None:
        _w,_h = frame.shape[1], frame.shape[0]
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        fps_out = fps_in / conf.SKIP_N
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter(record, fourcc, fps_out, (_w,_h))


    while cap.isOpened():
        ret, frame = cap.read()
        if frame_counter % conf.SKIP_N != 0:
            frame_counter += 1
            continue

        if not ret:
            break
        
        boxes, scores, labels = detector.detect_frame(frame)
        boxes, scores, labels = nms.filter_boxes(boxes, scores, labels)
        boxes, ids, labels = mot_tracker.update(boxes, scores, labels)
        frame = artist.draw(frame, boxes, ids, labels)


        if record is not None:
            video_writer.write(frame)
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        
        
        frame_counter += 1
        
        if not delay_done: input()
        delay_done = True

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()