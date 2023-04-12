import torch
from torch import nn
from model.base_model import BaseModel

class SampleModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(1,1)
    def forward(self, x):
        return self.Linear(x)