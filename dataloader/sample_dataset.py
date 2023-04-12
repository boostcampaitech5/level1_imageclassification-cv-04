from dataloader.base_dataset import BaseDataset
import pathlib
import torch
from util import split_dataset
from torch.utils.data import random_split
class SampleDataset(BaseDataset):
    def __init__(self,base_dir):
        super().__init__(base_dir)
        self.train_data = torch.FloatTensor([1,2,3,4,5])
        self.val_data = torch.FloatTensor([1,2,3])
    def __getitem__(self, idx):
        if self.train_mode:
            return torch.tensor([self.train_data[idx%5]]),torch.FloatTensor([idx])
        else:
            return torch.tensor([self.train_data[idx%3]]),torch.FloatTensor([idx])
    def _get_file_list(self):
        self.data = list(self.base_dir.glob('*'))

