import torch
from torch import nn
from abc import *

class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, loss):
        super().__init__()
        self.set_loss(loss)
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
    
    def set_loss(self,target_loss):
        loss_list = {"BCELoss":nn.BCELoss, "CrossEntropyLoss":nn.CrossEntropyLoss,
                    "MSELoss":nn.MSELoss,"Custom":self.custom_loss}
        self.criterion = loss_list[target_loss]
    def custom_loss(self,logit,y):
        raise NotImplementedError