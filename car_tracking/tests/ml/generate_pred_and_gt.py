"""
    Generate object detection predictions and ground truth for the car tracking dataset.
    In Pascal VOC format.
"""

import os
import cv2
import click

from ...datasets import AICityDataset
from ...detector import YOLOV5Detector


@click.command()
@click.argument('dataset_path')
@click.argument('dataset_split')
@click.argument('scenario_id')
@click.argument('output_path')
@click.argument('model_str')
def main(dataset_path, dataset_split, scenario_id, output_path, model_str):
    ds = AICityDataset(dataset_path, dataset_split)
    scenario = ds[scenario_id]

    detector = YOLOV5Detector(model_str, confidence=0.2, label_whitelist=[2,3,5,7])


    os.makedirs(os.path.join(output_path, "detections"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "groundtruths"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)

    for cam_id, cam in scenario.cameras.items():
        vid_path = cam.video_path
        cam_gt = cam.gt # Ground truth annotations  [frame_idx, id, x, y, w, h]
        cam_gt["frame"] = cam_gt["frame"].astype(int)
        cap = cv2.VideoCapture(vid_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gt = cam_gt[cam_gt["frame"] == frame_idx]

            if len(gt) == 0:
                frame_idx += 1
                continue

            gt_boxes = gt[["x", "y", "w", "h"]].values.astype(int)

            boxes, scores, labels = detector.detect_frame(frame)
            pred_boxes = boxes.astype(int)

            # Save detections and gts
            filename = f"{cam_id}_{frame_idx}.txt"
            with open(os.path.join(output_path, "detections", filename), "w") as f:
                for i,pred_box in enumerate(pred_boxes):
                    f.write("car ")
                    f.write(scores[i].astype(str) + " ")

                    f.write(f"{pred_box[0]} {pred_box[1]} {pred_box[2]} {pred_box[3]}\n")
            
            with open(os.path.join(output_path, "groundtruths", filename), "w") as f:
                for box in gt_boxes:
                    f.write("car ")
                    f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")
            
            im_path = os.path.join(output_path, "images", filename.replace(".txt", ".jpg"))
            cv2.imwrite(im_path, frame)
            frame_idx += 1
            print(frame_idx)


if __name__ == "__main__":
    main()