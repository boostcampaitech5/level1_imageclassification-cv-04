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
        image_name = self.df.iloc[idx].ImageID
        y=0
        if self.train:
            img = Image.open(self.df.iloc[idx].ImageID)
            label = self.df.iloc[idx].ans
            age = str(image_name).split('/')[-2].split('_')[-1]
        #  print(age)
            y = torch.zeros(3,dtype=torch.long)
            y[2]=(label)//6
            y[0]=label%6//3
            age = int(age)
            if age<30:
                y[1]=0
            elif age<60:
                y[1]=1
            else:
                y[1]=2
            #y[1]=age
        #   print(image_name,y,label)
        else:
            img_path = os.path.join(self.eval_path, 'images', self.df.iloc[idx].ImageID)
            img = Image.open(img_path)
        
        if self.transform:
            img = self.transform(img)

        return img, y


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = ClassificationDataset(csv_path = './input/data/train/train_info.csv',
                                    transform=transform)
    data_iter = DataLoader(dataset,
                           batch_size=3,
                           shuffle = True)
    print(dataset.df.columns)
    print(dataset.df['ImageID'])
  #  print(next(iter(data_iter)))
