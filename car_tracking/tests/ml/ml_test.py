import numpy as np
import pytest
from pathlib import Path
from ...detector import YOLOV5Detector
from ...tracker import SORT
from ...mtmc import DummyMTMC
from ...processor.mtmc import MTMCProcessor
import os

@pytest.fixture(scope="module")
def base_path() -> Path:
    """Get the root folder"""
    return Path(__file__).parent.parent.parent.parent

@pytest.fixture(scope="module")
def detector(base_path):
    curr_cwd = os.getcwd()
    os.chdir(base_path)
    det = YOLOV5Detector(model_str="yolov5n6", confidence=.2, label_whitelist=[2,3,5,7])
    os.chdir(curr_cwd)
    return det

@pytest.fixture(scope="module")
def tracker():
    return SORT()

@pytest.fixture(scope="module")
def mtmc():
    return DummyMTMC()

@pytest.fixture(scope="module")
def cameras_list():
    return ["cam_1", "cam_2", "cam_3"]

@pytest.fixture(scope="module")
def processor(detector, tracker, mtmc, cameras_list):
    trackers = {}


    for cam in cameras_list:
        trackers[cam] = tracker # shouldn't matter which tracker is used
    return MTMCProcessor(detector, trackers, mtmc)

def test_detector_shape(detector):
    test_frame = np.zeros((480, 640, 3))
    boxes, scores, labels = detector.detect_frame(test_frame)
    assert (boxes.shape[0] == scores.shape[0] == labels.shape[0]) and (boxes.shape[1] == 4)

def test_tracker_shape(detector, tracker):
    test_frame = np.zeros((480, 640, 3))
    boxes, scores, labels = detector.detect_frame(test_frame)
    boxes, ids, labels = tracker.update(boxes, scores, labels)
    
    assert (boxes.shape[0] == scores.shape[0] == labels.shape[0]) and (boxes.shape[1] == 4)

def test_mtmc_ids(processor, cameras_list):
    test_frames = {}
    clist = cameras_list # force pytest to eval fixture
    for cam in clist:
        test_frames[cam] = np.zeros((480, 640, 3))

    boxes_dict, ids_dict, labels_dict = processor.update(test_frames)
    assert len(boxes_dict) == len(ids_dict) == len(labels_dict) == len(cameras_list)



