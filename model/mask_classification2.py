import torch
from torch import nn
from model.base_model import BaseModel
import torch.nn.functional as F

from torchvision.models import resnet101


class MaskClassification2(BaseModel):
    def __init__(self,class_num, loss="MSELoss"):
        super().__init__(loss)
        self.class_num = class_num
        self.resnet = resnet101(pretrained=True)
        # for p in self.resnet.parameters():
        #     p.requires_grad = False
        self.resnet.layer4 = self.resnet.layer4[:1]
        self.resnet.fc = nn.Sequential(nn.Linear(2048,1024),
                                nn.Sigmoid(),
                                nn.Linear(1024,512),
                                nn.Sigmoid(),
                                nn.Linear(512,self.class_num)
                                )
      #  self.resnet.fc.requires_grad = True
        #print(self.resnet)
        
    def forward(self,x):
      #  print(x[0],x[1])
        x=self.resnet(x)
        return x
    
    
    def custom_loss(self,logit,y):
        ClassLoss = nn.CrossEntropyLoss()
        AGELoss = nn.MSELoss()
        loss = ClassLoss(logit[:,:2],y[:,0])+\
            AGELoss(logit[:,2:3],y[:,1:2].float())+\
            ClassLoss(logit[:,3:],y[:,2])
        
        return loss
    def accuracy(self,logit,label):
        gender = (logit[:,:2].argmax(dim=1) == label[:,0])
        boundary = torch.Tensor([30,60]).cuda()
        age = (torch.bucketize(logit[:,2],boundary)==torch.bucketize(label[:,1],boundary))
        mask = (logit[:,3:].argmax(dim=1)==label[:,2])
        all_correct = gender&age&mask
        return sum(all_correct)
