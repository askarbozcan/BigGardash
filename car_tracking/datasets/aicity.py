from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import pandas as pd
import os
from pathlib import Path

@dataclass
class Position:
    lat: float
    lon: float

@dataclass
class Camera:
    id: str
    timestamp: float
    gt: pd.DataFrame
    path: str
    position: Optional[Position] = None

    @property
    def video_path(self) -> str:
        return os.path.join(self.path, "vdo.avi")

@dataclass
class Scenario:
    id: str
    cameras: Dict[str, Camera]

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, key):
        return self.cameras[key]

class AICityDataset:
    VALID_SPLITS = ['train', 'validation', 'test']
    CAMERA_POSITIONS = {
        "c016": Position(42.4995215, -90.6906985),
        "c017": Position(42.4987505, -90.6906065),
        "default": Position(42.49880, -90.69069)
    }
    def __init__(self, path: str, split_type: str = 'train'):
        self.path = path
        self.split_type = split_type.lower()

        if split_type not in self.VALID_SPLITS:
            raise ValueError(f'Invalid split type: {split_type}')
        
        self.scenarios = {}

        self._load_data()

    def _load_data(self):
        root = Path(self.path)
        scenario_folder = root / self.split_type
        scenario_ids = [x.name for x in scenario_folder.iterdir() if x.is_dir()]

        # timestamps
        timestamps_for_cameras = {}
        timestamps_path = root / "cam_timestamp"
        for sid in scenario_ids:
            timestamps_file = timestamps_path / (sid + ".txt")
            with open(timestamps_file, "r") as f:
                line = f.readline().strip()
                while line != "":
                    cam_id, timestamp = line.split()
                    timestamps_for_cameras[cam_id] = float(timestamp)
                    line = f.readline().strip()

        for sid in scenario_ids:
            cameras = {}
            camera_ids = [x.name for x in (scenario_folder / sid).iterdir() if x.is_dir()]
            for cid in camera_ids:
                gt_path = (scenario_folder / sid / cid / "gt" / "gt.txt")
                gt = pd.read_csv(gt_path, sep=',', names=["frame", "id", "x", "y", "w", "h"], 
                                usecols=["frame", "id", "x", "y", "w", "h"])
                c = Camera(cid, timestamps_for_cameras[cid], gt, str(scenario_folder / sid / cid))
                try:
                    c.position = self.CAMERA_POSITIONS[cid]
                except KeyError:
                    c.position = self.CAMERA_POSITIONS["default"]
                    
                cameras[cid] = c

            s = Scenario(sid, cameras)
            self.scenarios[sid] = s

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        return self.scenarios[idx]

if __name__ == "__main__":
    ds = AICityDataset("/Users/askar/Documents/demoset", "train")
    print(ds["S04"].cameras["c016"].gt)




