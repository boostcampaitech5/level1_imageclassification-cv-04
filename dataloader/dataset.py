import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import pandas as pd
from PIL import Image
import os

class TrainDataset(Dataset):
    """Data loader를 만들기 위한 base dataset class"""

    def __init__(self, csv_path, transform=None, train=True):
        df = pd.read_csv(csv_path, dtype = {'ImageID':'str', 'ans':'str', 'type':'str'})
        if train:
            self.df = df[df['type'] == 'train']
        else:
            self.df = df[df['type'] == 'val']
        self.transform = transform


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx].ImageID)
        str_label = self.df.iloc[idx].ans
        label = int(str_label[0])*6 + int(str_label[1])*3 + int(str_label[2])

        if self.transform:
            img = self.transform(img)

        return img, torch.LongTensor([label]).squeeze()


class TestDataset(Dataset):
    """Data loader를 만들기 위한 base dataset class"""

    def __init__(self, csv_path, transform=None, eval_path=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.eval_path = eval_path


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        img_path = os.path.join(self.eval_path, 'images', self.df.iloc[idx].ImageID)
        img = Image.open(img_path)
        label = self.df.iloc[idx].ans

        if self.transform:
            img = self.transform(img)

        return img, torch.LongTensor([label]).squeeze()


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = TrainDataset(csv_path = '../input/data/train/train_info2.csv',
                                    transform=transform)
    data_iter = DataLoader(dataset,
                           batch_size=3,
                           shuffle = True)
    print(dataset.df.columns)
    print(dataset.df['ImageID'])
    print(next(iter(data_iter)))
