import numpy as np
from abc import ABC, ABCMeta, abstractmethod
from typing import List

class BaseCLIFS(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def match(self, frames: List, prompt: str):
        pass
