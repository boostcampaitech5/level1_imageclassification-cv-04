import os
import torch
from .transform import get_transform
from torchvision import transforms 
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import multiprocessing
from PIL import Image


class CustomDataset(Dataset):
    """Data loader를 만들기 위한 base dataset class"""

    def __init__(self, args:dict, train: bool = True):
        self.datadir = args.datadir
        self.train = train
        
        self.train_data = pd.read_csv(os.path.join(self.datadir, args.train_file))
        self.valid_data = pd.read_csv(os.path.join(self.datadir, args.valid_file))
        
        self.transform = args.transform


    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.valid_data)

    def __getitem__(self, idx):
        # Train Mode
        if self.train:
            img = Image.open(self.train_data.iloc[idx].ImageID)
            label = self.train_data.iloc[idx].ans
            if self.transform:
                transform, cfg = get_transform(self.transform)
                img = transform(img)
        
        # Validation Mode
        else:
            img = Image.open(self.valid_data.iloc[idx].ImageID)
            label = self.valid_data.iloc[idx].ans
            if self.transform:
                #이미지 사이즈 관련 Transform 만 진행
                val_trans = ["resize","centercrop","totensor","normalize"]
                trans_list = []
                for t in self.transform:
                    if t in val_trans:
                        trans_list.append(t)
                transform, cfg = get_transform(trans_list)
                img = transform(img)
    
        return img, torch.LongTensor([label]).squeeze()


def create_dataset(datadir: str, dataname: str, aug_name: str = 'default'):
    trainset = __import__('torchvision.datasets', fromlist='datasets').__dict__[dataname](
        root      = os.path.join(datadir,dataname), 
        train     = True, 
        download  = True, 
        transform = __import__('datasets').__dict__[f'{aug_name}_augmentation']()
    )
    testset = __import__('torchvision.datasets', fromlist='datasets').__dict__[dataname](
        root      = os.path.join(datadir,dataname), 
        train     = False, 
        download  = True, 
        transform = __import__('datasets').__dict__['test_augmentation']()
    )

    return trainset, testset


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = multiprocessing.cpu_count() // 2
    )
    
if __name__ == '__main__':
    import json
    import argparse
    
    parser = argparse.ArgumentParser(description="Classification for Computer Vision")
    with open('config.json') as f:
        config = json.load(f)

    for key in config:
        parser_key = key.replace('_', '-')
        parser.add_argument(f'--{parser_key}', default=config[key], type=type(config[key]))

    args = parser.parse_args()
    

    dataset = CustomDataset(args =args,train=True)
    data_iter = DataLoader(dataset,
                           batch_size=1,
                           shuffle = True)
    
    images, labels = next(iter(data_iter))
    
    print(images.shape)
    print(len(data_iter))