import eventlet
import socketio
import base64
import click
import cv2
from sympy import Q
from car_tracking.datasets.aicity import Scenario
from ..datasets import AICityDataset
from ..detector import YOLOV5Detector
from ..tracker import SORT
from ..mtmc import DummyMTMC
from ..processor.mtmc import MTMCProcessor

import json

sio = socketio.Server()
app = socketio.WSGIApp(sio)

PORT = 4920
SKIP_N = 5


def mtmc_generator(scenario, processor):
    boxes_dict, ids_dict, labels_dict = {}, {}, {}
    while True:
        caps = {}
        frame_counters = {}
        for cam in scenario.cameras.values():
            caps[cam.id] = cv2.VideoCapture(cam.video_path)
            frame_counters[cam.id] = 0

        videos_finished = False
        while not videos_finished:
            frames_dict = {}

            for cam_id, cap in caps.items():

                if cap.isOpened():
                    ret, frame = cap.read()
                    if frame_counters[cam_id] % SKIP_N != 0:
                        frame_counters[cam_id] += 1
                        continue

                    if not ret:
                        videos_finished = True
                        break

                    frames_dict[cam_id] = frame
                    frame_counters[cam_id] += 1
            

            boxes_dict, ids_dict, labels_dict = processor.update(frames_dict)
            if len(boxes_dict) == 0:
                continue

            yield {"frames": frames_dict, "boxes": boxes_dict, "ids": ids_dict, "labels": labels_dict}


def get_current_data_dict(results_generator):
    for result in results_generator:

        frames = {}
        for cam_id, frame in result["frames"].items():
            jpeg = cv2.imencode(".JPEG", frame)[1].tobytes()
            encoded = base64.b64encode(jpeg)
            frames[cam_id] = str(encoded)

        boxes = {cam_id: x.tolist() for cam_id, x in result["boxes"].items()}
        labels = {cam_id: x.tolist() for cam_id, x in result["labels"].items()}
        response_dict = {"frames":frames, "boxes": boxes, "ids": result["ids"], "labels": labels}
        yield response_dict

@sio.event
def give_stream_data(sid):
    ds = AICityDataset(app.config["dataset_path"], app.config["dataset_split"])
    scenario = ds[app.config["scenario_id"]]

    detector = YOLOV5Detector(model_str="yolov5s6", confidence=.2, label_whitelist=[2,3,5,7])
    trackers = {}
    for cam in scenario.cameras.values():
        trackers[cam.id] = SORT()
    mtmc = DummyMTMC()
    processor = MTMCProcessor(detector, trackers, mtmc)

    generator = get_current_data_dict(mtmc_generator(scenario, processor))
    for data_dict in generator:
        sio.emit("receive_stream_data", data_dict)

@click.command()
@click.option("--port", default=PORT, help="Port to run the server on.")
@click.option("--dataset_path", help="AI City dataset path")
@click.option("--dataset_split", default="train", help="AI City dataset split")
@click.option("--scenario_id", help="AI City scenario id")
def main(port, dataset_path, dataset_split, scenario_id):
    app.config["dataset_path"] = dataset_path
    app.config["dataset_split"] = dataset_split
    app.config["scenario_id"] = scenario_id
    app.run(host="0.0.0.0", port=port, debug=True)

if __name__ == "__main__":
    main()
