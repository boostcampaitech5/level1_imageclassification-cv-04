import torch
from torch import nn
from model.base_model import BaseModel

class MaskClassification(BaseModel):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(10,10)
    def forward(self,x):
        pass
    def custom_loss(self,logit,y):
        CELoss = nn.CrossEntropyLoss()
        MSELoss = nn.MSELoss()

        loss = CELoss(logit[:2],y[:2])+\
            MSELoss(logit[2:3],y[2:3])+\
            CELoss(logit[3:],y[3:])