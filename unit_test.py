from dataloader.dataset import ClassificationDataset
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader,Subset

import pandas as pd
import numpy as np
import torch
from utils.sampler import *

#sklearn train_test_split동작확인
def split_test():
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = ClassificationDataset(csv_path = './input/data/train/train_info.csv',
                                    transform=transform)
    print('-'*10)
    print(dataset.df.columns)
    print(dataset.df['ImageID'].to_numpy().shape)
    print(dataset.df['ans'].to_numpy().shape)

    train_idx, val_idx = train_test_split(np.arange(len(dataset))
                                        ,train_size=0.8
                                        ,stratify=dataset.df['ans']
                                        ,random_state=223
                                        )

    train_set = Subset(dataset,train_idx)
    val_set = Subset(dataset,val_idx)
    print(len(train_set),len(val_set))
    df_cnt = dataset.df['ans'].value_counts().sort_index()
    train_cnt = dataset.df['ans'][train_idx].value_counts().sort_index()
    val_cnt = dataset.df['ans'][val_idx].value_counts().sort_index()
    print(pd.concat([df_cnt,train_cnt,val_cnt],axis=1))
    dataloader = DataLoader(train_set)
    #print(next(iter(dataloader)))

#weighted_sampler 동작 확인
def w_sampler_test():
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = ClassificationDataset(csv_path = './input/data/train/train_info.csv',
                                    transform=transform)
    print('-'*10)
    print(dataset.df.columns)
    print(dataset.df['ImageID'].to_numpy().shape)
    print(dataset.df['ans'].to_numpy().shape)

    train_set,val_set,train_idx,val_idx = train_valid_split_by_sklearn(dataset,223)
    sampler = weighted_sampler(dataset,train_idx,18)
    sampler2 = weighted_sampler(dataset,val_idx,18)
    loader = DataLoader(train_set,sampler=sampler)
    class_list = torch.zeros(18)
    for x,y in loader:
        class_list[y]+=1
    print(class_list)
if __name__ == '__main__':
    #split_test()
    w_sampler_test()