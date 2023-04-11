import torch
from torch import nn
from abc import *

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
    @abstractmethod
    def __forward(self, x):
        raise NotImplementedError