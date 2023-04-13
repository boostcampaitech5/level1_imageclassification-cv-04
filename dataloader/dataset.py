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


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        if self.train:
            img = Image.open(self.df.iloc[idx].ImageID)
        else:
            img_path = os.path.join(self.eval_path, 'images', self.df.iloc[idx].ImageID)
            img = Image.open(img_path)
        label = self.df.iloc[idx].ans

        if self.transform:
            img = self.transform(img)

        return img, torch.LongTensor([label]).squeeze()


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = ClassificationDataset(csv_path = './input/data/eval/info.csv',
                                    transform=transform,
                                    num_classes = 18)
    data_iter = DataLoader(dataset,
                           batch_size=3,
                           shuffle = True)
    
    print(next(iter(data_iter)))
