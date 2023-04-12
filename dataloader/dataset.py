
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils import data
import pandas as pd
from PIL import Image
import os
import csv

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
         
        y = self.Y[idx]
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
        
if __name__ == '__main__':
    TRAIN_IMG_DIR = "/opt/ml/input/data/train/"
    dataset = IC_Dataset(TRAIN_IMG_DIR)

    dataloader = data.DataLoader(dataset,batch_size=64)
    train_features, train_labels = next(iter(dataloader))
    print(train_labels[0])