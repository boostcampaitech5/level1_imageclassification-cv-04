from dataloader.base_dataset import BaseDataset
import pathlib
import torch
from util import split_dataset
from torch.utils.data import random_split
from PIL import Image
import PIL
from torchvision import transforms
import pandas as pd

class MaskDataset3(BaseDataset):
    def __init__(self,base_dir, transform = None):
        super().__init__(base_dir)
        self.train_dir = self.base_dir/'train'
        self.val_dir = self.base_dir/'eval'
        self.train_data_dir = self.train_dir/'images'
        self.val_data_dir = self.val_dir/'images'
        
        self.train_data = self._get_file_list(self.train_data_dir)
        self.val_data = self._get_file_list(self.val_data_dir)
        self.val_csv = pd.read_csv(self.val_dir/'info.csv').set_index('ImageID')
        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((232,232)),
                                    transforms.CenterCrop((224,224)),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomRotation(90),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                    ])
        else:
            self.transform = transform

        print('sample data')
        print(f'train_set len: {len(self.train_data)}')
        print(f'valid_set len: {len(self.val_data)}')
        
    def __getitem__(self, idx):
        if self.train_mode:
            target_data = self.train_data
        else:
            target_data = self.val_data
        
        data_dir = target_data[idx]
        pil_data = Image.open(data_dir)
        data = self.transform(pil_data)
        if self.train_mode:
            annotation = self.parse_annotation(data_dir)
            y = self.annotation_to_tensor(annotation)
            return data,y
        else:
            data_id = str(data_dir).split('/')[-1]
            
            return data,data_id

    def __len__(self):
        return super().__len__()
    
    def _get_file_list(self,target_dir):
        if self.train_mode:
            data_list = list(target_dir.glob('**/[mask|normal|incorrect]*'))
        else:
            data_list = list(target_dir.glob('**/*'))
        return data_list
    def parse_annotation(self,file_dir):
        annotation = file_dir.parent.name
        file_name = file_dir.name
        gender,nation,age = annotation.split('_')[1:]
        mask_type = file_name.split('.')[0]
        #print(annotation,file_name)
        #print(gender,nation,age,mask_type)
        return [gender, nation, age, mask_type]
    
    def annotation_to_tensor(self,annotation):
        gender = annotation[0]
        age = int(annotation[2])
        mask_type = annotation[3]
        #(male, female, age, mask, normal, incorrect)
        idx = 0
        
        if age<30:
            idx+=0
        elif age<60:
            idx+=1
        else:
            idx+=2

        if gender == 'male':
            idx+= 0
        elif gender == 'female':
            idx+=3
        else:
            raise ValueError(f"gender not in [male,female]: {gender}")
        
        if mask_type.startswith('mask'):
            idx+=0
        elif mask_type.startswith('incorrect'):
            idx+=6
        elif mask_type.startswith('normal'):
            idx+=12
        else:
            raise ValueError(f"File name not in [mask,normal,incorrect]: {mask_type}")
        return idx
    def class_to_tensor(self,num):
        answer = torch.zeros(1).long()
        
        return answer