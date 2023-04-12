from torch.utils.data import Dataset
from abc import *
import pathlib
class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, base_dir):
        print('Data dir: ',base_dir)
        self.base_dir = pathlib.Path(base_dir)
        self.data_list = self._get_file_list()
        self.train_data = []
        self.val_data = []
        self.train_mode = True
    def __len__(self):
        if self.train_mode:
            return len(self.train_data)
        else:
            return len(self.val_data)
    
    @abstractmethod
    def __getitem__(self,idx):
        raise NotImplementedError
    @abstractmethod
    def _get_file_list(self):
        raise NotImplementedError
    def set_train_mode(self):
        self.train_mode = True
    def set_val_mode(self):
        self.train_mode = False
