from torch.utils.data import Dataset
from abc import *
import pathlib
class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, base_dir):
        print('Data dir: ',base_dir)
        self.base_dir = pathlib.Path(base_dir)
        self.data_list = self._get_file_list()
    def __len__(self):
        return len(self.data)
    
    @abstractmethod
    def __getitem__(self,idx):
        raise NotImplementedError
    @abstractmethod
    def _get_file_list(self):
        raise NotImplementedError

