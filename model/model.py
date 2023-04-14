import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()

        self.num_classes = args.num_classes
        self.load_model = args.load_model
        if self.load_model:
            self.backbone = load_backbone(self.load_model)
            for name, param in self.backbone.named_parameters():
                for layer_name in args.not_freeze_layer:
                    if name.find(layer_name) == -1:
                        param.requires_grad = False
        self.backbone.fc = nn.Linear(2048, 1000)

        self.fc = nn.Sequential(nn.Linear(1000, 512),
                                nn.Dropout(0.5),
                                nn.Linear(512, 128),
                                nn.Linear(128, self.num_classes))
        

    def forward(self, x):
        if self.load_model:
            x = self.backbone(x)
        x = self.fc(x)
        return x


def load_backbone(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

    return model


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
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # print(summary(model.to('cuda:0'), (3, 256, 256)))
    # img = torch.randn(3, 3, 224, 224)
    # output = model(img)
    # print(output.shape)

