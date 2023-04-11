from dataloader.base_dataset import BaseDataset
import pathlib
import torch
from util import split_dataset
from torch.utils.data import random_split
class SampleDataset(BaseDataset):
    def __init__(self,base_dir):
        super().__init__(base_dir)
    def __getitem__(self, idx):
        return torch.FloatTensor([idx]),torch.FloatTensor([idx])
    def _get_file_list(self):
        self.data = list(self.base_dir.glob('*'))

def make_dataset(dataset_config):
    dataset = SampleDataset(**dataset_config['args'])
    ratio = split_dataset(dataset_config['split_ratio'],len(dataset))
    train,val = random_split(dataset,ratio)
    print('Train dataset: ',len(train))
    print('Val dataset: ',len(val))
    return train,val