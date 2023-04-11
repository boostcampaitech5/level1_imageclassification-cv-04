import torch
from torch.utils.data import dataloader

import importlib


def get_loader(loader_name, args):
    target_lib = 'dataloader.'+loader_name
    print("Target Module: ",target_lib)
    loader_lib = importlib.import_module(target_lib)
    target_loader = "".join(map(str.title,target_lib.split('_')))
    print("Target Loader: ",target_loader)
    for name,cls in loader_lib.__dict__:
        print(name,cls)

