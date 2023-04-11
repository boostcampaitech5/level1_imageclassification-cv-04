import torch
from torch import nn

import dataloader


class Trainer():
    def __init__(self,model,data_dir,config):
        self.name= None
        self.model = model
        self.dataset = dataloader.get_loader('sample_loader',{})

    def train():
        pass
