import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset, DataLoader,Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def weighted_sampler(dataset, num_classes):
    class_counts = [0] * num_classes
    labels = []
    for _, label in dataset:
        class_counts[label] += 1
        labels.append(label)
    num_samples = len(dataset)

    class_weights = [num_samples / class_counts[i] for i in range(num_classes)] 

    # 해당 데이터에 해당하는 class 가중치
    weights = [class_weights[labels[i]] for i in range(num_samples)]
    sampler = WeightedRandomSampler(torch.FloatTensor(weights), num_samples)

    return sampler

def train_valid_split_by_sklearn(dataset):
    train_idx, val_idx = train_test_split(np.arange(len(dataset))
                                      ,train_size=0.8
                                      ,stratify=dataset.df['ans']
                                      ,random_state=223
                                      )
    train_set = Subset(dataset,train_idx)
    val_set = Subset(dataset,val_idx)
    return train_set,val_set