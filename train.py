import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import torchvision.transforms as transforms


from model.model import AlexNet

from dataloader.dataset import IC_Dataset


########################### Define ##################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device >> {device}")

seed = torch.initial_seed()

NUM_EPOCHS = 90 
BATCH_SIZE = 128
MOMENTUM = 0.9
LR_DECAY = 0.0005
LR_INIT = 0.01
IMAGE_DIM = 227 
NUM_CLASSES = 18  
DEVICE_IDS = [0]
TRAIN_IMG_DIR = "/opt/ml/input/data/train/images/"
LABEL_PATH = "/opt/ml/input/data/train/train.csv"

#####################################################################

transform_list = [transforms.CenterCrop(IMAGE_DIM),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]

alexnet = AlexNet(num_classes=NUM_CLASSES).to(device)

dataset = IC_Dataset(TRAIN_IMG_DIR,transform_list)
print('Dataset created')

dataloader = data.DataLoader(
        dataset,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        batch_size=BATCH_SIZE)
print('Dataloader created')

optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
print('Optimizer created')

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) #일정 step 마다 lr 에 gamma 곱함
print('LR Scheduler created')


print('Starting training...')
total_steps = 1
for epoch in range(NUM_EPOCHS):
    for imgs, classes in dataloader:
        imgs, classes = imgs.to(device), classes.to(device)
        
        output = alexnet(imgs)
        loss = F.cross_entropy(output, classes)
        
        # update the parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log the information and add to tensorboard
        if total_steps % 10 == 0:
            with torch.no_grad():
                _, preds = torch.max(output, 1)
                accuracy = torch.sum(preds == classes)

                print('Epoch: {} \tStep: {} \tLoss: {:.4f} \tAcc: {}'
                    .format(epoch + 1, total_steps, loss.item(), accuracy.item()/BATCH_SIZE))

        # print out gradient values and parameter average values
        if total_steps % 100 == 0:
            with torch.no_grad():

                print('*' * 10)
                for name, parameter in alexnet.named_parameters():
                    if parameter.grad is not None:
                        avg_grad = torch.mean(parameter.grad)
                        print('\t{} - grad_avg: {}'.format(name, avg_grad))

                    if parameter.data is not None:
                        avg_weight = torch.mean(parameter.data)
                        print('\t{} - param_avg: {}'.format(name, avg_weight))
  
        total_steps += 1
    lr_scheduler.step()