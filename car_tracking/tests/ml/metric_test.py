from email.mime import base
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
