
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os


class CD_Dataset(Dataset):
    def __init__(self,img_dir,trans_list):
        self.dataset_dir = img_dir
        self.classes = os.listdir(self.dataset_dir)
        self.transform = trans_list
        X_list = []
        y_list = []

        for idx, c in enumerate(self.classes):
            name = os.listdir(self.dataset_dir + f"\\{c}")
            for i in name:
                X_list.append(self.dataset_dir + f"\\{c}\\" + i)
            y_list = y_list + [idx for i in range(len(name))]
            

        self.X = X_list
        self.y = y_list
        
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        X = Image.open(self.X[idx])
        trans_X = transforms.Compose(self.transform)(X)
        y = self.y[idx]
        return trans_X,y 