import click
import cv2
import motmetrics as mm

from ...datasets import AICityDataset
from ...detector import YOLOV5Detector
from ...tracker import SORT

@click.command()
@click.argument("dataset_path")
@click.argument("dataset_split")
@click.argument("scenario_id")
@click.argument("model_str")
def main(dataset_path, dataset_split, scenario_id, model_str):
    ds = AICityDataset(dataset_path, dataset_split)
    scenario = ds[scenario_id]

    detector = YOLOV5Detector(model_str, confidence=0.2, label_whitelist=[2,3,5,7])

    summaries = []
    for cam_id, cam in scenario.cameras.items():
        tracker = SORT()
        metric_acc = mm.MOTAccumulator(auto_id=False)

        cap = cv2.VideoCapture(cam.video_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if cam.gt[cam.gt["frame"] != frame_idx].shape[0] == 0:
                frame_idx += 1
                continue

            gt_boxes = cam.gt[["x", "y", "w", "h"]].values.astype(int)
            gt_ids = cam.gt["id"].values.astype(int)
            boxes, scores, labels = detector.detect_frame(frame)
            boxes, ids, labels = tracker.update(boxes, scores, labels)

            pred_boxes = boxes.astype(int)
            pred_ids = ids.astype(int)
            
            dists = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
            metric_acc.update(gt_ids, pred_ids, dists, frame_idx)

            frame_idx += 1
            print(frame_idx)
        
        mh = mm.metrics.create()
        summary = mh.compute(metric_acc, metrics=['num_frames', 'mota', 'motp', "idf1", "idp", "idr"], name='acc')
        summaries.append(summary)
    
    total_summary = sum(summaries)/len(summaries)
    print(summaries)

if __name__ == "__main__":
    main()