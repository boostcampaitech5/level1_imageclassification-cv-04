import torch
import numpy as np
import random
import json

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
def read_json(config):
    with open(config,'r') as f:
        json_file = json.load(f)
    return json_file
def write_json(dir,data):
    with open(dir,'w') as f:
        json.dump(data, f,indent = 2)

def split_dataset(ratio,total_len):
    train = int(total_len*ratio)
    return [train,total_len-train]