import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

class Classifier(nn.Module):
    def __init__(self, num_classes, load_model=None):
        super(Classifier, self).__init__()

        self.num_classes = num_classes
        self.load_model = load_model
        if load_model:
            self.backbone = load_backbone(load_model)
            # for name, param in self.backbone.named_parameters():
            #     param.requires_grad = False
        self.backbone.fc = nn.Linear(2048, 1000)

        self.fc = nn.Sequential(nn.Linear(1000, 512),
                                nn.Dropout(0.5),
                                nn.Linear(512, 128),
                                nn.Linear(128, num_classes))
        

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
    model = Classifier(num_classes=18, load_model='resnet50')
    print(summary(model.to('cuda:0'), (3, 256, 256)))
    # img = torch.randn(3, 3, 224, 224)
    # output = model(img)
    # print(output.shape)

