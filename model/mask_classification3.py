import torch
from torch import nn
from model.base_model import BaseModel
import torch.nn.functional as F

from torchvision.models import resnet101


class MaskClassification3(BaseModel):
    def __init__(self,class_num, loss="MSELoss"):
        super().__init__(loss)
        self.class_num = class_num
        self.resnet = resnet101(pretrained=True)
        # for p in self.resnet.parameters():
        #     p.requires_grad = False
        self.resnet.layer4 = self.resnet.layer4[:1]
        self.resnet.layer4.requires_grad = True
        self.resnet.fc = nn.Sequential(nn.Linear(2048,1024),
                                nn.BatchNorm1d(1024),
                                nn.LeakyReLU(0.1),
                                nn.Dropout(0.2),
                                nn.Linear(1024,512),
                                nn.BatchNorm1d(512),
                                nn.Sigmoid(),
                                nn.Linear(512,self.class_num)
                                )
        self.resnet.fc.requires_grad = True
        self.sigmoid = nn.Sigmoid()
        print(self.resnet)
        
    def forward(self,x):
      #  print(x[0],x[1])
        x=self.resnet(x)
        x[:,:2]=self.sigmoid(x[:,:2])
        x[:,3:]=self.sigmoid(x[:,3:])
        return x
    
    
    def custom_loss(self,logit,y):
        ClassLoss = nn.CrossEntropyLoss()
        AGELoss = nn.MSELoss()
        loss = ClassLoss(logit,y)
        
        return loss
    def accuracy(self,logit,label):
        
        cor = (logit.argmax(dim=1)==label)
        
        
        return sum(cor)
    def convert_inference(self,logit):
        g=logit[:,:2].argmax(dim=1)
        boundary = torch.Tensor([30,60]).cuda()
        a=torch.bucketize(logit[:,2],boundary)
        m=logit[:,3:].argmax(dim=1)
        return g,a,m