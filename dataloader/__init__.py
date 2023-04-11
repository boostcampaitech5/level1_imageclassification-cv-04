import torch
from torch.utils.data import dataloader,random_split

import importlib


def get_loader(config):
    print('Load Dataset start')
    dataset_name = config['name']
    target_lib = 'dataloader.'+dataset_name
    print("Target Module: ",target_lib)
    dataset_lib = importlib.import_module(target_lib)

    if config['already_split']:
        pass
        target_train_dataset = ''.join(map(str.title,dataset_name.split('_')))
        print("Target train dataset: ",target_dataset,config)
        train_dataset = getattr(dataset_lib,target_dataset)(**config)
        print("Target valid dataset: ",)
    else:
        target_dataset = ''.join(map(str.title,dataset_name.split('_')))
        print("Target dataset: ",target_dataset)
        dataset = getattr(dataset_lib,target_dataset)(**config['args'])
        split_ratio = config['split_ratio']
        print("Automatically split. ratio: ",split_ratio)
        train_len = int(len(dataset)*split_ratio)
        train_set,valid_set = random_split(dataset, [train_len,len(dataset)-train_len])
        print("Train_len: ",len(train_set))
        print("Valid_len: ",len(valid_set))
        train_loader = dataloader.DataLoader(train_set,config['dataloader'])
        valid_loader = dataloader.DataLoader(valid_set,config['dataloader'])
    return train_set,valid_set

