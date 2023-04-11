import torch
from torch.utils.data import dataloader,random_split

import importlib


def get_dataset(config):
    print('Load Dataset start')
    dataset_name = config['name']
    target_lib = 'dataloader.'+dataset_name
    print("Target Module: ",target_lib)
    dataset_lib = importlib.import_module(target_lib)

    if config['train']:
        if config['already_split']:
            print("Exist train and valid dataset")
            train_config = config['train_set']
            valid_config = config['valid_set']
            target_train = ''.join(map(str.title,train_config['name'].split('_')))
            valid_train = ''.join(map(str.title,valid_config['name'].split('_')))
            print("Target train dataset: ",target_train,
                  train_config['args']['base_dir'])
            print("Target valid dataset: ",valid_train,
                  valid_config['args']['base_dir'])
            train_set = getattr(dataset_lib,train_config['name'])(**train_config['args'])
            valid_set = getattr(dataset_lib,train_config['name'])(**valid_config['args'])
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
        return train_set,valid_set
    else:
        pass

def get_loader(config, dataset_list):
    loaders = []
    for dataset in dataset_list:
        loaders.append(dataloader.DataLoader(dataset,**config))
    return loaders