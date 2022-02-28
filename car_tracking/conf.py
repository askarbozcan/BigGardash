import os
from pathlib import Path
ROOT_FOLDER = Path(os.path.realpath(__file__)).parent.parent.absolute()
ROOT_FOLDER = str(ROOT_FOLDER)

try:
    DARKNET_PATH = os.environ["DARKNET_PATH"]
except KeyError:
    raise NameError("DARKNET_PATH not found in environment variables. Please point it to where darknet is compiled.\
                    \nex (Linux): export DARKNET_PATH=/path/to/darknet")

GPU_ENABLED = False
ENABLE_DELAY_INIT = False

SKIP_N = 5
SEQ_NMS_N = 20
ALLOWED_CLASSES = ["car", "bus", "bicycle", "motorcycle"]
ALLOWED_CLASSES_IDX = {v:i for i,v in enumerate(ALLOWED_CLASSES)}
SEQ_NMS_THRESH = 0.05
SEQ_NMS_NMS_THRESH = 0.5
YOLO_THRESH = 0.05


RAND_COLORS = []
import random
for i in range(10):
    RAND_COLORS.append((random.randint(100,255), random.randint(100,250), random.randint(100,250)))