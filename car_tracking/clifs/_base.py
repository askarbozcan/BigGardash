import numpy as np
from abc import ABC, ABCMeta, abstractmethod
from typing import List,Dict

class BaseCLIFS(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def match(self, frames: Dict[str, np.ndarray], boxes: Dict[str, np.ndarray],\
                    ids: Dict[str, np.ndarray], labels: Dict[str, np.ndarray],
                    prompt: str):
        pass
