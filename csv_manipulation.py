import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import pandas as pd
from PIL import Image
import os

# data_dir = '../input/data/train/'
# img_dir = f'{data_dir}/images'
# df2_path = f'{data_dir}/train.csv'

# df2 = pd.read_csv(df2_path)
# print(df2.head())

csv_path = 'kfold4.csv'
df = pd.read_csv(csv_path)
print(df.head())

for index, row in df.iterrows():
    # print(row['ImageID'], row['ImageID'].split('/')[-2].split('_')[-1])    
    df['age'] = row['ImageID'].split('/')[-2].split('_')[-1]

print(df.head())