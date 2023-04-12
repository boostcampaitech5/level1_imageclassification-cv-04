import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import pandas as pd
from PIL import Image

class ClassificationDataset(Dataset):
    """Data loader를 만들기 위한 base dataset class"""

    def __init__(self, csv_path, transform=None, num_classes=-1):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.num_classes = num_classes


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx].ImageID)
        label = self.df.iloc[idx].ans

        if self.transform:
            img = self.transform(img)

        return img, torch.LongTensor([label])
    
if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = ClassificationDataset(csv_path = './input/data/train/train_info.csv',
                                    transform=transform,
                                    num_classes = 18)
    data_iter = DataLoader(dataset,
                           batch_size=3,
                           shuffle = True)
    
    print(next(iter(data_iter)))
