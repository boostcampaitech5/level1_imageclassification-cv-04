from dataloader.base_loader import BaseLoader
import pathlib

class SampleLoader(BaseLoader):
    def __init__(self,base_dir):
        super().__init__(base_dir)
    def __getitem__(self, idx):
        print('get item')
    def _get_file_list(self):
        print('get_file_list')