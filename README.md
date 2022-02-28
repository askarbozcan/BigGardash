## Car Tracking **Preliminary** Version

_For API docs, go here https://github.com/GlobalMaksimum/traffic-car-detection/blob/feature/docs/car_tracking/README.md._

### Easy setup
1) Run easy_setup.py as such, specifying into which folder Darknet should be installed:

`python easy_setup.py -p /folder/to/clone/darknet`

2) Define DARKNET_PATH environment variable so that inference script will be aware of where darknet is located:

`export DARKNET_PATH=/path/to/darknet`

3) Done.


### (Alternative) Manual Setup
1) Clone darknet repo into a directory of your choice (https://github.com/AlexeyAB/darknet/)

`git clone https://github.com/AlexeyAB/darknet/`

2) edit top of Makefile found in root of darknet as such:

```
GPU=0
CUDNN=0
CUDNN_HALF=0
OPENCV=0
AVX=1
OPENMP=0
LIBSO=1
ZED_CAMERA=0
ZED_CAMERA_v2_8=0
```
(Note that this is the tried config for compilation of CPU-only darknet on Mac which seems to cause no issues.)

3) Call "make" inside darknet root folder.

**Note:** If any linker error related "libdarknet.so" pops up, it is OK as long as at the end "libdarknet.so" can be found in the root directory.

4) Add *DARKNET_PATH* to your environment variables.

`export DARKNET_PATH=/path/to/darknet/root/`

3) Call `download_yolo_pretrained.py` which will download all predefined YOLO models and configs (or just one of them) as such:
```
python download_yolo_pretrained.py -f /path/to/pretrained_models

//Alternatively (to just download yolov4-tiny):
python download_yolo_pretrained.py -f /path/to/pretrained_models -n yolov4-tiny

```

4) While inside root folder of traffic-car-detection (where car_tracking folder is located) of the package, call:
```
python fix_yolo_paths.py -f /path/to/pretrained_models

python setup.py build_ext --inplace
```
to compile necessary (for SeqNMS) Cython package and to fix directories for YOLO configs. 

*(YOLO configs are unable to parse relative dirs, hence the need to have such a script.)*


5) Done.

### CLI Usage with visualisation
---
To perform inference using CLI api, simply run the command below while in the root folder where "car_tracking" folder is located.

```
python -m car_tracking.video_main_modular -v "./path/to/video" -r "./path/to/save/detection/video
```
This will show a window of tracking of a video and save the result as mp4 to path pointed to by -r.






### Python API usage examples
---

#### Using VideoProcessor
Processor is a class which combines data reading, timestamp collection and detection + tracking into one.

```python

## EXPORT DARKNET_PATH before running

from os.path import join as pjoin
from car_tracking.detector import YOLODetector
from car_tracking.nms import SeqNMS
from car_tracking.tracker import SORT
from car_tracking.processor import VideoProcessor

# initialize models
model_path = "/path/to/yolo/config&weights"
detector = YOLODetector(pjoin(model_path,"yolov4-tiny.cfg"), 
                        pjoin(model_path, "coco.data"), 
                        pjoin(model_path, "yolov4-tiny.weights"),
                        thresh=0.05)

nms = SeqNMS(frame_N=5) # buffer for seqnms    
mot_tracker = SORT()

# glue them all together
vid_processor = VideoProcessor(detector, nms, mot_tracker)
vid_processor.set_video("/path/to/video/to/infer")


for timestamp, boxes, ids, labels in vid_processor.process():
	""" 
    	do what is needed with the data here
    	
        timestamp: float, representing time elapsed since the beginning of the video
        boxes: np.array[N*4], bounding boxes of objects in format (x1,y1,x2,y2)
        ids: np.array[N], Unique ids of objects as assigned by tracker
        labels: np.array[N], Class labels of objects
    """
    pass
 

```



#### (Alternative) YOLO + SeqNMS + SORT
```python

## EXPORT DARKNET_PATH before running

from os.path import join as pjoin
from car_tracking.detector import YOLODetector
from car_tracking.nms import SeqNMS
from car_tracking.tracker import SORT

model_path = "/path/to/yolo/config&weights"
detector = YOLODetector(pjoin(model_path,"yolov4-tiny.cfg"), 
                        pjoin(model_path, "coco.data"), 
                        pjoin(model_path, "yolov4-tiny.weights"),
                        thresh=0.05)

nms = SeqNMS(frame_N=5) # buffer for seqnms    
mot_tracker = SORT()

frame = .... # read a frame [H*W*3] np.array 

# output after detector
boxes, scores, labels = detector.detect_frame(frame) 

# filtered boxes from detector
boxes, scores, labels = nms.filter_boxes(boxes, scores, labels) 

# tracked boxes from tracker (only makes sense in sequential input)
boxes, ids, labels = mot_tracker.update(boxes, scores, labels) 
```
