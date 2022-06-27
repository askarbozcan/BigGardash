import eventlet
from matplotlib.image import thumbnail
import socketio
import base64
import click
import cv2
from car_tracking.datasets.aicity import Scenario
from ..datasets import AICityDataset
from ..detector import YOLOV5Detector
from ..tracker import SORT
from ..mtmc import DummyMTMC
from ..processor.mtmc import MTMCProcessor
from ..clifs import DummyCLIFS, CLIPBG
import json
import random
import numpy as np

sio = socketio.Server(async_handlers=False, cors_allowed_origins="*", engineio_logger=False)
app = socketio.WSGIApp(sio)

PORT = 4920
SKIP_N = 2

class MTMCGeneration:
    last_generated_dict: dict = None

    def __init__(self, dataset_path, dataset_split, scenario_id):
        ds = AICityDataset(dataset_path, dataset_split)
        self.scenario = ds[scenario_id]

        self.detector = YOLOV5Detector(model_str="yolov5s6", confidence=.4, label_whitelist=[2,3,5,7])
        self.trackers = {}
        for cam in self.scenario.cameras.values():
            self.trackers[cam.id] = SORT(min_hits=2, max_age=4, iou_threshold=.2)
        self.mtmc = DummyMTMC()
        self.processor = MTMCProcessor(self.detector, self.trackers, self.mtmc)



    def mtmc_generator(self):
        boxes_dict, ids_dict, labels_dict = {}, {}, {}
        while True:
            caps = {}
            frame_counters = {}
            for cam in self.scenario.cameras.values():
                caps[cam.id] = cv2.VideoCapture(cam.video_path)
                frame_counters[cam.id] = 0

            videos_finished = False
            while not videos_finished:
                frames_dict = {}

                for cam_id, cap in caps.items():

                    if cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            videos_finished = True
                            break

                        #frame = cv2.resize(frame, (640, 480))
                        if frame_counters[cam_id] % SKIP_N != 0:
                            frame_counters[cam_id] += 1
                            continue

                        if not ret:
                            videos_finished = True
                            break

                        frames_dict[cam_id] = frame
                        frame_counters[cam_id] += 1
                

                boxes_dict, ids_dict, labels_dict = self.processor.update(frames_dict)
                if len(boxes_dict) == 0:
                    continue
                
                # generate thumbnails
                thumbnails = {}
                for cam_id, frame in frames_dict.items():
                    cutouts = []
                    ignored_idxs = []
                    for i,box in enumerate(boxes_dict[cam_id]):
                        box = box.astype(int)
                        cutout = frame[box[1]:box[3], box[0]:box[2]]
                        if cutout.shape[0] < 1 or cutout.shape[1] < 1:
                            ignored_idxs.append(i)
                            continue
                        cutouts.append(cutout)

                    thumbnails[cam_id] = cutouts
                    boxes_dict[cam_id] = np.delete(boxes_dict[cam_id], ignored_idxs, axis=0)
                    ids_dict[cam_id] = [ids_dict[cam_id][i] for i in range(len(ids_dict[cam_id])) if i not in ignored_idxs]
                    labels_dict[cam_id] = np.delete(labels_dict[cam_id], ignored_idxs, axis=0)

                res = {"frames": frames_dict, "boxes": boxes_dict, "ids": ids_dict, "labels": labels_dict, "thumbnails":thumbnails}
                MTMCGeneration.last_generated_dict = res
                yield res

    def get_current_data_dict(self):
        for result in self.mtmc_generator():

            frames = {}
            for cam_id, frame in result["frames"].items():
                frame = cv2.resize(frame, (600, 480), interpolation=cv2.INTER_AREA)
                frames[cam_id] = self.convert_frame_to_jpeg(frame)
            
            thumbnails = {}
            for cam_id, cutouts in result["thumbnails"].items():
                thumbnails[cam_id] = [self.convert_frame_to_jpeg(cutout) for cutout in cutouts]
            

            boxes = {cam_id: x.tolist() for cam_id, x in result["boxes"].items()}
            labels = {cam_id: x.tolist() for cam_id, x in result["labels"].items()}
            response_dict = {"frames":frames, "boxes": boxes, "ids": result["ids"], "labels": labels, "thumbnails": thumbnails}
            yield response_dict

    @staticmethod
    def convert_frame_to_jpeg(frame):
        jpeg = cv2.imencode(".JPEG", frame)[1].tobytes()
        encoded = base64.b64encode(jpeg)
        return encoded.decode("utf-8")


#########
global_generator: MTMCGeneration = None # defined in __main__
global_data_queue = eventlet.queue.Queue(maxsize=5)

@sio.event
def give_car_positions(sid):
    print("Received request for car positions")
    data_dict = global_generator.last_generated_dict
    positions = []

    random.seed(492)
    for cam_id, cam in global_generator.scenario.cameras.items():
        for id_ in data_dict["ids"][cam_id]:
           rand_lat = cam.position.lat + random.normalvariate(0, 0.0001)
           rand_lon = cam.position.lon + random.normalvariate(0, 0.0001)
           positions.append({"car_id": id_, "lat": rand_lat, "lon": rand_lon, "cam_id": cam_id})
    
    return positions


@sio.event
def give_stream_data(sid, cam_id):
    data_dict = global_data_queue.get()
    response_dict = {}
    for k in data_dict.keys():
        response_dict[k] = data_dict[k][cam_id]

    #print(response_dict)
    print("Sending data for cam_id: {}".format(cam_id))
    
    random.seed(492)
    car_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(data_dict["ids"]))]
    response_dict["car_colors"] = car_colors

    response_list = []
    for i in range(len(response_dict["ids"])):
        item = {
            "box": response_dict["boxes"][i],
            "id": response_dict["ids"][i],
            "label": response_dict["labels"][i],
            "thumbnail": response_dict["thumbnails"][i],
        }

        response_list.append(item)

    response_dict_new = {
        "frame": response_dict["frames"],
        "cars": response_list
    }
    #sio.emit("receive_stream_data", data_dict)
    return response_dict_new
@sio.event
def give_camera_info(sid):
    print("accepted connection camera_info")
    if global_generator.last_generated_dict is None:
        return {}

    cameras = {}
    for cam_id, cam in global_generator.scenario.cameras.items():
        thumbnail = global_generator.last_generated_dict["frames"][cam_id]
        thumbnail = cv2.resize(thumbnail, (400, 240), interpolation=cv2.INTER_AREA)
        thumbnail = global_generator.convert_frame_to_jpeg(thumbnail)
        cameras[cam_id] = {"lat": cam.position.lat, "lon": cam.position.lon, "thumbnail": thumbnail}
    
    return cameras

#clifs_matcher = DummyCLIFS()
clifs_matcher = CLIPBG()

@sio.event
def give_match_info(sid, prompt: str):
    print("accepted connection match_info")
    
    if global_generator.last_generated_dict is None:
        return []
    
    data_dict = global_generator.last_generated_dict
    matches = clifs_matcher.match(data_dict["frames"], data_dict["boxes"], data_dict["ids"], data_dict["labels"], prompt)
    for i,m in enumerate(matches):
        matches[i]["frame"] = MTMCGeneration.convert_frame_to_jpeg(m["frame"])
    
    print("Finished matching")
    #sio.emit("receive_match_info", matches)
    return matches


def threaded_model():
    generator = global_generator.get_current_data_dict()

    while True:
        for data_dict in generator:
            print(global_data_queue.qsize(), "Queue size")
            if global_data_queue.full():
                global_data_queue.get(block=True)
            global_data_queue.put(data_dict, block=True)

            sio.sleep(0.1)
            print("Finished iteration model")

@click.command()
@click.option("--port", default=PORT, help="Port to run the server on.")
@click.option("--dataset_path", help="AI City dataset path")
@click.option("--dataset_split", default="train", help="AI City dataset split")
@click.option("--scenario_id", help="AI City scenario id")
def main(port, dataset_path, dataset_split, scenario_id):
    global global_generator
    global_generator = MTMCGeneration(dataset_path, dataset_split, scenario_id)
    
    eventlet.spawn(threaded_model)
    eventlet.wsgi.server(eventlet.listen(('', port)), app)

if __name__ == "__main__":
    main()
