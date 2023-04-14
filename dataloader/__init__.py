import torch
from torch.utils.data import dataloader,random_split
import importlib




def get_loader(config, dataset):
    return dataloader.DataLoader(dataset,**config)


def convert_dataset_name(name):
    name = "".join(map(str.title,name.split("_")))
    return name
                   
def split_dataset(dataset,split_ratio):
    set_len = len(dataset)
    train_len = int(set_len*split_ratio)
    val_len = set_len-train_len
    train,val = random_split(dataset,[train_len,val_len])
    return train,val
'''
def get_dataset(config):
    print('Load Dataset start')
    dataset_name = config['name']
    target_lib = 'dataloader.'+dataset_name
    print("Target Module: ",target_lib)
    dataset_lib = importlib.import_module(target_lib)

    target_dataset = ''.join(map(str.title,config['name'].split('_')))
    print("Target dataset: ",target_dataset)
    dataset = getattr(dataset_lib,target_dataset)(**config['args'])
    print("Dataset_len: ",len(dataset))
    return dataset
    '''
