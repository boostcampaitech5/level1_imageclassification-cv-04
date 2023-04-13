from dataloader.base_dataset import BaseDataset
import pathlib
import torch
from util import split_dataset
from torch.utils.data import random_split
from PIL import Image
import PIL
from torchvision import transforms
import pandas as pd

val_dir = pathlib.Path('../input2/data')/'eval'
val_data_dir = val_dir/'images'
data_list = list(val_data_dir.glob('**/[a-z0-9]*'))

val_csv = pd.read_csv(val_dir/'info.csv')
data_dir = str(data_list[0])
print('------------')
print(data_dir)

data_dir = data_dir.split('/')[-1]
print(data_dir)
val_csv = val_csv.set_index('ImageID')
#print(val_csv.head(10))
print('******')
print(val_csv.loc[data_dir]['ans'])