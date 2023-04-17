import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
from timm import create_model, list_models

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.num_classes = args.num_classes
        self.load_model = args.load_model
        if self.load_model:
            # list_models('resnet*', pretrained=True)
            self.backbone1 = create_model('resnet50', pretrained=True)
            self.backbone2 = create_model('resnet50', pretrained=True)
            self.backbone3 = create_model('resnet50', pretrained=True)
        # for p in self.backbone.parameters():
        #     p.requires_grad = False
        self.backbone1 = nn.Sequential(*list(self.backbone1.children())[:-1])
        self.backbone2 = nn.Sequential(*list(self.backbone2.children())[:-1])
        self.backbone3 = nn.Sequential(*list(self.backbone3.children())[:-1])

            
        self.batch = nn.BatchNorm1d(2048)

        self.gender_fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024,2)
        )
        self.age_fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024,1)
        )
        self.mask_fc = nn.Sequential(
            nn.BatchNorm1d(2048),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024,3)
        )
        
    def forward(self, x):
        if self.load_model:
            x1 = self.backbone1(x)
            x2 = self.backbone2(x)
            x3 = self.backbone3(x)
       # x1=self.batch(x1)
        gender = self.gender_fc(x1)
        age = self.age_fc(x2)
        mask = self.mask_fc(x3)
        logit = torch.cat((gender,age,mask),dim=1)
        return logit


if __name__ == '__main__':
    args_dict = {'seed' : 223,
                'csv_path' : './input/data/train/train_info.csv',
                'save_path' : './checkpoint',
                'use_wandb' : True,
                'wandb_exp_name' : 'exp4',
                'wandb_project_name' : 'Image_classification_mask',
                'wandb_entity' : 'connect-cv-04',
                'num_classes' : 18,
                'model_summary' : True,
                'batch_size' : 64,
                'learning_rate' : 1e-4,
                'epochs' : 100,
                'train_val_split': 0.8,
                'save_mode' : 'model',
                'save_epoch' : 10,
                'load_model':'resnet50',
                'transform_path' : './transform_list.json',
                'transform_list' : ['resize', 'randomhorizontalflip', 'randomrotation', 'totensor', 'normalize'],
                'not_freeze_layer' : ['layer4']}
    
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    model = Classifier(args)
    #print(model)
    #print(model.load_model.resnet50)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # print(summary(model.to('cuda:0'), (3, 256, 256)))
    # img = torch.randn(3, 3, 224, 224)
    # output = model(img)
    # print(output.shape)