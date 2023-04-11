from dataloader.base_dataset import BaseDataset
import pathlib

class SampleDataset(BaseDataset):
    def __init__(self,base_dir):
        super().__init__(base_dir)
    def __getitem__(self, idx):
        print('get item')
    def _get_file_list(self):
        self.data = list(self.base_dir.glob('*'))