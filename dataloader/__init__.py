import torch
from torch.utils.data import dataloader

import importlib


def get_loader(loader_name, args):
    target_lib = 'dataloader.'+loader_name
    print("Target Module: ",target_lib)
    loader_lib = importlib.import_module(target_lib)
    target_loader = ''.join(map(str.title,loader_name.split('_')))
    print("Target Loader: ",target_loader)
    getattr(loader_lib,target_loader)(**args)

