import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
from PIL import Image


from model.models import AlexNet
from model.model_finetune import fineTune

from dataloader.dataset import IC_Dataset, IC_Test_Dataset
import sys

NUM_EPOCHS = 10
BATCH_SIZE = 10
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
LR = 0.0001
IMAGE_DIM = 384 
NUM_CLASSES = 18  
DEVICE_IDS = [0]
TRAIN_IMG_DIR = "/opt/ml/input/data/eval/"
OUT_PATH = "./level1_imageclassification-cv-04/data_out/model_save/"
MODEL = "Resnet18"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device >> {device}")

transform_list = [transforms.CenterCrop(IMAGE_DIM),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]

model = fineTune(models.resnet18(pretrained=True).to(device), MODEL, NUM_CLASSES).to(device)

model.load_state_dict(torch.load(OUT_PATH + "resnet18_80.pt"))
model.eval()

test_dataset = IC_Test_Dataset(TRAIN_IMG_DIR,transform_list)

test_dataloader = data.DataLoader(
        test_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)


df = pd.read_csv(TRAIN_IMG_DIR + "info.csv")

cnt = 0
for x,y in test_dataloader:
    print(cnt,end="\r")
    cnt += 1
    x = x.to(device)
    out = model(x)

    for i in range(len(y)):
        predict = torch.argmax(out[i])
        df.loc[df["ImageID"] == y[i],"ans"] = predict.item()

df.to_csv("./level1_imageclassification-cv-04/data_out/test_1.csv",index = False)
