import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import pandas as pd
from PIL import Image
import os

class ClassificationDataset(Dataset):
    """Data loader를 만들기 위한 base dataset class"""

    def __init__(self, csv_path, transform=None, train=True, eval_path=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.train = train
        self.eval_path = eval_path
        self.labels = self.df['ans']
        self.img_path = self.df['ImageID']


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        if self.train:
            img = Image.open(self.img_path[idx])
        else:
            img_path = os.path.join(self.eval_path, 'images', self.img_path[idx])
            img = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, torch.LongTensor([label]).squeeze()
    

class KFoldDataset(Dataset):
    """Data loader를 만들기 위한 base dataset class"""

    def __init__(self, csv_path, kfold=-1, train=True, transform=None):
        df = pd.read_csv(csv_path)
        if train:
            self.df = df[df['fold'] != kfold]
        else:
            self.df = df[df['fold']==kfold]
        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx].ImageID)
        label = self.df.iloc[idx].ans

        if self.transform:
            img = self.transform(img)

        return img, label


class KFoldSplitDataset(Dataset):
    """Data loader를 만들기 위한 base dataset class"""

    def __init__(self, csv_path, kfold=-1, train=True, transform=None, split=None):
        df = pd.read_csv(csv_path)
        if train:
            self.df = df[df['fold'] != kfold]
        else:
            self.df = df[df['fold']==kfold]
        self.transform = transform
        self.split = split


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx].ImageID)
        label = self.df.iloc[idx].ans
        if self.split == 'mask':
            if 0 <= label < 6:
                label = 0
            elif 6 <= label < 12:
                label = 1
            else:
                label = 2
        elif self.split == 'gender':
            if label%6 < 3:
                label = 0
            else:
                label = 1
        elif self.split == 'age':
            if label % 3 == 0:
                label = 0
            elif label % 3 == 1:
                label = 1
            else:
                label = 2

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = KFoldSplitDataset(csv_path = '../input/data/train/kfold4.csv',
                                    transform=transform,
                                    kfold=0, train=True)
    
    data_iter = DataLoader(dataset,
                           batch_size=3,
                           shuffle = True)
    cnt=0
    for _, y in data_iter:
        print(y)
        break
    print(dataset.df.columns)
    print(dataset.df['ImageID'])
  #  print(next(iter(data_iter)))
