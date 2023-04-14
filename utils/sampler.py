import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader,Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def weighted_sampler(dataset, data_idx, num_classes):

    target_df=dataset.df.loc[data_idx]

    class_counts = target_df['ans'].value_counts().sort_index().to_numpy()
    labels = target_df['ans'].to_numpy()
    num_samples = len(dataset)

    class_weights = num_samples / class_counts 

    # 해당 데이터에 해당하는 class 가중치
    weights = class_weights[labels]



    #log
    # print(class_counts)
    # print(type(class_counts))
    # print(labels)
    # print(len(labels),len(weights))
    # print(weights)
    sampler = WeightedRandomSampler(torch.FloatTensor(weights), num_samples)

    return sampler

def train_valid_split_by_sklearn(dataset,seed=223):
    train_idx, val_idx = train_test_split(np.arange(len(dataset))
                                      ,train_size=0.8
                                      ,stratify=dataset.df['ans']
                                      ,random_state=seed
                                      )
    train_set = Subset(dataset,train_idx)
    val_set = Subset(dataset,val_idx)
    return train_set,val_set, train_idx, val_idx