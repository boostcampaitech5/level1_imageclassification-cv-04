
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
import csv

#Mask Image Classification 마스크 착용 여부
class IC_Dataset(Dataset): 
    def __init__(self,img_dir, trans_list = None):
        self.dataset_dir = img_dir
        self.transform = trans_list
        
        
        self.files = self.getXY(self.dataset_dir)
        
        
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        X = Image.open(self.X[idx])
        trans_X = transforms.Compose(self.transform)(X)
        y = self.y[idx]
        return trans_X,y 
    
    #각 폴더에 있는 이미지들의 경로 생성 후 Class id 까지
    def getXY(self,root_path):
        X = []
        Y = []
        f = open(root_path + "train_info.csv")
        reader = csv.reader(f)
        for line in reader:
            print(line)
            X.append(line[0])
            Y.append(line[1])
            
            
        
        return X,Y
        
        
TRAIN_IMG_DIR = "/opt/ml/input/data/train/"
dataset = IC_Dataset(TRAIN_IMG_DIR)
print(dataset)