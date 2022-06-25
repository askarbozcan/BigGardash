import numpy as np
from ._base import BaseCLIFS
from typing import Dict
class DummyCLIFS(BaseCLIFS):
    def __init__(self):
        pass

    def match(self, frames: Dict[str, np.ndarray], boxes: Dict[str, np.ndarray],\
                    ids: Dict[str, np.ndarray], labels: Dict[str, np.ndarray],
                    prompt: str):

        dummy_response1 = {
            "frame": np.zeros((300,300,3), dtype=np.uint8),
            "camera": "c017",
            "id": "dummy_id"
        }
        dummy_response2 = {
            "frame": np.zeros((300,300,3), dtype=np.uint8),
            "camera": "c016",
            "id": "dummy_id2"
        }

        return [dummy_response1, dummy_response2]