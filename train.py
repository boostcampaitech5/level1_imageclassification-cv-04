import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

from model.models import AlexNet
from model.model_finetune import fineTune

from dataloader.dataset import IC_Dataset
import sys

import wandb

########################### Define ##################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device >> {device}")

seed = torch.initial_seed()

NUM_EPOCHS = 80
BATCH_SIZE = 64
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
LR = 0.0001
IMAGE_DIM = 384 
NUM_CLASSES = 18  
DEVICE_IDS = [0]
TRAIN_IMG_DIR = "/opt/ml/input/data/train/"
OUT_PATH = "./level1_imageclassification-cv-04/data_out/model_save/"
MODEL = "resnet18"

Test_name= f"exp1_bs{BATCH_SIZE}_ep{NUM_EPOCHS}_adam_lr{LR}_{MODEL}.jh"

wandb.init(name = Test_name,
            project = 'Image_classification_mask',
            entity = 'connect-cv-04',
            )
wandb.config = {'model': MODEL, 'lr': LR, 'epochs' : NUM_EPOCHS, 'batch_size' : BATCH_SIZE }
#####################################################################

transform_list = [transforms.CenterCrop(IMAGE_DIM),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]

model = fineTune(models.resnet18(pretrained=True).to(device), MODEL, NUM_CLASSES).to(device)
# model = models.resnet18(pretrained=True).to(device)
# model = model(3,18)

# print(summary(model,(3,IMAGE_DIM,IMAGE_DIM),device = "cuda"))
# sys.exit()

dataset = IC_Dataset(TRAIN_IMG_DIR,transform_list)

train_dataset , val_dataset = random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

if (not os.path.isdir(OUT_PATH + Test_name)):
	os.makedirs(OUT_PATH + Test_name)

print('Dataset created')

train_dataloader = data.DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)

val_dataloader = data.DataLoader(
        val_dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)


print('Dataloader created')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LR)
print('Optimizer created')

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) #일정 step 마다 lr 에 gamma 곱함
print('LR Scheduler created')


print('Starting training...')
total_steps = 1
for epoch in range(NUM_EPOCHS):
    model.train()
    for imgs, classes in train_dataloader:
        imgs, classes = imgs.to(device), classes.to(device).long()
        
        output = model(imgs)
        # print(classes.shape)
        # print(output[0])
        loss = criterion(output, classes)
        
        # update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(output, 1)
        accuracy = torch.sum(preds == classes)

        
    model.eval()
    for v_imgs, v_classes in val_dataloader:
        v_imgs, v_classes = v_imgs.to(device), v_classes.to(device).long()
        with torch.no_grad():
            val_out = model(v_imgs)
            
        val_loss = criterion(val_out, v_classes)
        _, val_preds = torch.max(val_out, 1)
        val_accuracy = torch.sum(val_preds == v_classes)
        
    print('Epoch: {} \tStep: {} \tTrain_Loss: {:.4f} \tTrain_Acc: {} \tVal_Loss: {:.4f} \tVal_Acc: {}'
        .format(epoch + 1, total_steps, loss.item(), accuracy.item()/BATCH_SIZE,
                val_loss.item(), val_accuracy.item()/BATCH_SIZE))
    
    wandb.log({
        "Train Acc": 100. * accuracy.item()/BATCH_SIZE,
        "Train Loss": loss.item(),
        "Val Acc": 100. * val_accuracy.item()/BATCH_SIZE,
        "Val Loss": val_loss.item()})
                
            
    total_steps += 1
    lr_scheduler.step()
        
    ################### Wandb 로 로그 기록 ##################

    if epoch % 5 == 0:
        path = os.path.join(OUT_PATH + Test_name + f"{MODEL}_{epoch:02}.pt")
        torch.save(model.state_dict(), path)
    