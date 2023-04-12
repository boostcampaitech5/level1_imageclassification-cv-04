import torch
from torch import nn
from model.base_model import BaseModel

class MaskClassification(BaseModel):
    def __init__(self,loss=nn.CrossEntropyLoss):
        super().__init__(loss)
        self.conv1 = nn.Conv2d(3,8,3,1,1)
        self.conv2 = nn.Conv2d(8,32,3,1,1)
        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.conv4 = nn.Conv2d(64,128,3,1,1)
        self.conv5 = nn.Conv2d(128,256,3,1,1)
        self.fc1 = nn.Linear(12544,1024)
        self.fc2 = nn.Linear(1024,6)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x=self.relu(self.pool(self.conv1(x)))
        x=self.relu(self.pool(self.conv2(x)))
        x=self.relu(self.pool(self.conv3(x)))
        x=self.relu(self.pool(self.conv4(x)))
        x=self.relu(self.pool(self.conv5(x)))
        x=torch.flatten(x,1)
        x=self.sigmoid(self.fc1(x))
        x=self.fc2(x)
        return x
    def custom_loss(self,logit,y):
        CELoss = nn.CrossEntropyLoss()
        MSELoss = nn.MSELoss()
        loss = CELoss(logit[:,:2],y[:,:2])+\
            MSELoss(logit[:,2:3],y[:,2:3])+\
            CELoss(logit[:,3:],y[:,3:])
        return loss