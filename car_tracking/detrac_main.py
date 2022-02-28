from copy import Error
import numpy as np
from os.path import join as pjoin
from collections import deque
import cv2
import sys
import motmetrics as mm
from numpy.core.fromnumeric import sort
from .import conf

from .datasets import DETRACDataset
import click

sys.path.append(conf.DARKNET_PATH)

from .detector import YOLODetector
from .nms import SeqNMS
from .tracker import SORT



@click.command()
@click.option("-s","--seq_name", help="Sequence name")
@click.option("-d", "--dataset_path", help="Detrac dataset path")
@click.option("-i", "--show_ignored", help="Show ignored?", default=False, show_default=True)
@click.option("-r", "--record", help="Record path", default=None, show_default=True)


def main(seq_name, dataset_path, show_ignored, record):
    raise NotImplementedError("Detrac processing was not ported to modular yet.")

    delay_done = not conf.ENABLE_DELAY_INIT
    

    model_path = "./pretrained_models/yolov4-tiny"
    detector = YOLODetector(pjoin(model_path,"yolov4-tiny.cfg"), 
                            pjoin(model_path, "coco.data"), 
                            pjoin(model_path, "yolov4-tiny.weights"),
                            thresh=conf.YOLO_THRESH,
                            allowed_classes=conf.ALLOWED_CLASSES)

    nms = SeqNMS(conf.SEQ_NMS_N, conf.SEQ_NMS_THRESH, conf.SEQ_NMS_NMS_THRESH)
    mot_tracker = SORT()
    dataset = DETRACDataset(dataset_path, seq_name)

    metric_acc = mm.MOTAccumulator(auto_id=True) # accumulator

    if record is not None:
        _w,_h = dataset[0][0].shape[1], dataset[0][0].shape[0]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_writer = cv2.VideoWriter(record, fourcc, 8, (_w,_h))

    frame_counter = 0
    avg_fps = 0
    for frame, gt_boxes, gt_ids in dataset:
        if frame_counter % conf.SKIP_N != 0:
            frame_counter += 1
            continue

        boxes, scores, labels = detector.detect_frame(frame)
        boxes, scores, labels = nms.filter_boxes(boxes, scores, labels)
        boxes, ids, labels = mot_tracker.update(boxes, scores, labels)

        processed_det = []
        for label, confidence, bbox in det:
            processed_det.append((label, confidence, adjusted_bbox))

    
        
        # compute metrics
        pred_ids = sort_boxes[:, -1].copy().astype(int)
        pred_boxes = sort_boxes[:, :-1].copy()
        pred_boxes[:, 2] -= pred_boxes[:, 0] # convert x2 to width
        pred_boxes[:, 3] -= pred_boxes[:, 1] # convert y2 to height
        

        to_take = []
        for i,b in enumerate(pred_boxes):
            if not dataset.in_ignored((b[0]+b[2]/2, b[1]+b[3]/2)):
                to_take.append(i)

        pred_boxes = pred_boxes[to_take, :] # ignore "ignored" boxes
        pred_ids = pred_ids[to_take]

        gt_boxes = np.array(gt_boxes)# already in x,y,w,h format
        dists = mm.distances.iou_matrix(pred_boxes, gt_boxes, max_iou=0.5)
        metric_acc.update(np.array(gt_ids), pred_ids, dists)

        ## draw detections
        blk = np.zeros(frame.shape, np.uint8)


        for b in sort_boxes:
            _id = int(b[-1])

            color_i = _id % len(conf.RAND_COLORS)

            color = conf.RAND_COLORS[color_i]
            cv2.putText(frame, str(_id), (int(b[0]+10), int(b[1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,0,0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 3)
            cv2.rectangle(blk, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, cv2.FILLED)


        frame = cv2.addWeighted(frame, 1.0, blk, 0.25, 1)

        # draw ignored regions
        if show_ignored:
            ignored_reg = np.zeros(frame.shape, np.uint8)
            for ir in dataset.ignored_regions:
                ig_box = [ir["left"], ir["top"], ir["left"]+ir["width"], ir["top"]+ir["height"]]
                cv2.rectangle(frame, (int(ig_box[0]), int(ig_box[1])), (int(ig_box[2]), int(ig_box[3])), (50,50,50), cv2.FILLED)
                # cv2.rectangle(frame, (int(ig_box[0]), int(ig_box[1])), (int(ig_box[2]), int(ig_box[3])), (0,0,0), 3)
            

        if record is not None:
            video_writer.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break        


        frame_counter += 1
        
        if not delay_done: input()
        delay_done = True

    if record is not None:
        video_writer.release()
        
    cv2.destroyAllWindows()

    mh = mm.metrics.create()
    summary = mh.compute(metric_acc, metrics=['num_frames', 'mota', 'motp', "idf1"], name='acc')
    print(summary)
    print("Average FPS:", avg_fps*conf.SKIP_N/frame_counter)
    
if __name__ == "__main__":
    main()