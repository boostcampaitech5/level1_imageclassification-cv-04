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
            self.backbone = create_model(self.load_model, pretrained=True, num_classes=args.num_classes)

    def forward(self, x):
        if self.load_model:
            x = self.backbone(x)
        return x


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
                'load_model':'resnet18',
                'transform_path' : './transform_list.json',
                'transform_list' : ['resize', 'randomhorizontalflip', 'randomrotation', 'totensor', 'normalize'],
                'not_freeze_layer' : ['layer4']}
    
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    model = Classifier(args)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # print(summary(model.to('cuda:0'), (3, 256, 256)))
    # img = torch.randn(3, 3, 224, 224)
    # output = model(img)
    # print(output.shape)