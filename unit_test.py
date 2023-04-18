from dataloader.dataset import ClassificationDataset
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader,Subset
import torchvision as tv
import pandas as pd
import numpy as np
import torch
from utils.sampler import *
import matplotlib.pyplot as plt
#sklearn train_test_split동작확인
def split_test():
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = ClassificationDataset(csv_path = './input/data/train/train_info.csv',
                                    transform=transform)
    print('-'*10)
    print(dataset.df.columns)
    print(dataset.df['ImageID'].to_numpy().shape)
    print(dataset.df['ans'].to_numpy().shape)
    print(dataset.df['ImageID'][::7])
    print(dataset.df['ans'][::7].value_counts())
    train_idx, val_idx = train_test_split(7*np.arange(len(dataset)//7)
                                        ,train_size=0.8
                                        ,stratify=dataset.df['ans'][::7]
                                        ,random_state=223
                                        )
    train_idx=train_idx+np.arange(7).reshape(-1,1)
    val_idx=val_idx+np.arange(7).reshape(-1,1)
    train_idx=train_idx.reshape(-1,)
    val_idx=val_idx.reshape(-1,)
    train_set = Subset(dataset,train_idx)
    val_set = Subset(dataset,val_idx)
    print(len(train_set),len(val_set))
    df_cnt = dataset.df['ans'].value_counts().sort_index()
    train_cnt = dataset.df['ans'][train_idx].value_counts().sort_index()
    val_cnt = dataset.df['ans'][val_idx].value_counts().sort_index()
    print(pd.concat([df_cnt,train_cnt,val_cnt],axis=1))
    dataloader = DataLoader(train_set)
    #print(next(iter(dataloader)))

def sklearn_sampling_test():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ClassificationDataset(csv_path = './input/data/train/train_info.csv',
                                    transform=transform)
    train_set,val_set,train_idx,val_idx = train_valid_split_by_sklearn(dataset, train_ratio=0.8, seed=223)
    print(len(train_set),len(val_set))
    print(train_idx.shape)
    print(len(np.unique( np.concatenate((train_idx,val_idx))  )))
    print(len(np.unique(train_idx//7)))
    print(val_idx//7)
    df_cnt = dataset.df['ans'].value_counts().sort_index()
    train_cnt = dataset.df['ans'][train_idx].value_counts().sort_index()
    val_cnt = dataset.df['ans'][val_idx].value_counts().sort_index()
    print(pd.concat([df_cnt,train_cnt,val_cnt],axis=1))
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
    weighted_sampler(dataset,train_idx,18)
if __name__ == '__main__':
    # split_test()
    sklearn_sampling_test()
   # w_sampler_test()

    # transform = transforms.Compose([transforms.Normalize((0.45,0.45,0.45),(0.15,0.15,0.15)),transforms.Grayscale(3)])
    # transform2=transforms.Compose([transforms.Normalize((0.445,0.445,0.445),(0.35,0.35,0.35)),transforms.Grayscale(3)])
    # transform3=transforms.Compose([transforms.Grayscale(3)])
    # dataset = ClassificationDataset(csv_path = './input/data/train/train_info.csv',
    #                                 transform=transforms.ToTensor())
    # dataloader= DataLoader(dataset,shuffle=True)
    # import time
    # for idx,img in enumerate(dataloader):
    #     img=img[0]
    #     img1=transform(img*1.2).numpy().squeeze()
    #     img2=transform2(img*1.2).numpy().squeeze()
    #     img3=transform3(img).numpy().squeeze()
    #     plt.imshow(np.transpose(img.numpy().squeeze(),(1,2,0)))
    #     plt.savefig(f'0image.png')
    #     plt.imshow(np.transpose(img1,(1,2,0)))
    #     plt.savefig(f'1image.png')
    #     plt.imshow(np.transpose(img2,(1,2,0)))
    #     plt.savefig(f'2image.png')
    #     plt.imshow(np.transpose(img3,(1,2,0)))
    #     plt.savefig(f'3image.png')
    #     plt.imshow(np.transpose((img1+img2+img3)/3,(1,2,0)))
    #     plt.savefig(f'4image.png')
    #     break