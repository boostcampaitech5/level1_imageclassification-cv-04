
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torch.utils import data
import pandas as pd
from PIL import Image
import os
import csv
import sys

#Mask Image Classification 마스크 착용 여부
class IC_Dataset(Dataset): 
    def __init__(self,img_dir, trans_list = [transforms.ToTensor()]):
        self.dataset_dir = img_dir
        self.transform = trans_list
        
        
        x,y = self.getXY(self.dataset_dir)
        self.X = x
        self.Y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        X = Image.open(self.X[idx])
        trans_X = transforms.Compose(self.transform)(X)
         
        y = float(self.Y[idx])
        # print(torch.Tensor(y))
        return trans_X,y
    
    #각 폴더에 있는 이미지들의 경로 생성 후 Class id 까지
    def getXY(self,root_path):
        x = []
        y = []
        csv_path = os.path.join(root_path , "train_info.csv")
        f = open(csv_path)
        reader = csv.reader(f)
        next(iter(reader))
        for line in reader:
            x.append(line[0])
            y.append(line[1])
            
        return x,y
    
class IC_Test_Dataset(Dataset): 
    def __init__(self,img_dir, trans_list = [transforms.ToTensor()]):
        self.dataset_dir = img_dir
        self.transform = trans_list
        
        x= self.getXY(self.dataset_dir)
        self.X = x
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        x = os.path.join(self.dataset_dir , "images", self.X[idx])
        X = Image.open(x)
        trans_X = transforms.Compose(self.transform)(X)

        return trans_X,self.X[idx]
    
    #각 폴더에 있는 이미지들의 경로 생성 후 Class id 까지
    def getXY(self,root_path):
        x = []
        csv_path = os.path.join(root_path , "info.csv")
        f = open(csv_path)
        reader = csv.reader(f)
        next(iter(reader))
        for line in reader:
            x.append(line[0])
        return x
    
        
if __name__ == '__main__':
    TRAIN_IMG_DIR = "/opt/ml/input/data/eval/"
    dataset = IC_Test_Dataset(TRAIN_IMG_DIR)

    dataloader = data.DataLoader(dataset,batch_size=1)
    for imgs in dataloader:
        print(imgs)
        sys.exit()