import numpy as np
from ._base import BaseCLIFS

class DummyCLIFS(BaseCLIFS):
    def __init__(self):
        pass

    def match(self, frames, prompt):
        dummy_match1 = {"id": "dummy1", "location": "cam0", "frame": np.zeros((360, 360, 3), dtype=np.uint8)}
        dummy_match2 = {"id": "dummy2", "location": "cam0", "frame": np.zeros((360, 360, 3), dtype=np.uint8)}

        return [dummy_match1, dummy_match2]