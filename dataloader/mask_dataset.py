from dataloader.base_dataset import BaseDataset
import pathlib
import torch
from util import split_dataset
from torch.utils.data import random_split
from PIL import Image
import PIL
from torchvision import transforms

class MaskDataset(BaseDataset):
    def __init__(self,base_dir, transform = None):
        super().__init__(base_dir)
        self.train_dir = self.base_dir/'train'
        self.val_dir = self.base_dir/'eval'
        self.train_data_dir = self.train_dir/'images'
        self.val_data_dir = self.val_dir/'images'
        
        self.train_data = self._get_file_list(self.train_data_dir)
        self.valid_data = self._get_file_list(self.val_data_dir)

        if transform == None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((232,232)),
                                    transforms.CenterCrop((224,224)),
                                    transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                                    ])
        else:
            self.transform = transform

        print('sample data')
        print(self.train_data[0])
        print(self.parse_annotation(self.train_data[0]))
    def __getitem__(self, idx):
        if self.train_mode:
            target_data = self.train_data
        else:
            target_data = self.val_data
        
        data_dir = target_data[idx]
        pil_data = Image.open(data_dir)
        data = self.transform(pil_data)
        annotation = self.parse_annotation(data_dir)
        y = self.annotation_to_tensor(annotation)

        return data,y
    def __len__(self):
        return super().__len__()
    
    def _get_file_list(self,target_dir):
        data_list = list(target_dir.glob('**/[mask|normal|incorrect]*'))
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
        age = annotation[2]
        mask_type = annotation[3]
        #(male, female, age, mask, normal, incorrect)
        answer = torch.zeros(6,dtype=torch.float)
        if gender == 'male':
            answer[0] = 1
        elif gender == 'female':
            answer [1] = 1
        else:
            raise ValueError(f"gender not in [male,female]: {gender}")
        answer[2] = float(age)
        if mask_type.startswith('mask'):
            answer[3] = 1
        elif mask_type.startswith('normal'):
            answer[4] = 1
        elif mask_type.startswith('incorrect'):
            answer[5] = 1
        else:
            raise ValueError(f"File name not in [mask,normal,incorrect]: {mask_type}")
        return answer