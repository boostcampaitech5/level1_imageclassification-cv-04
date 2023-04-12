import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        x = nn.Flatten()(x)
        x = nn.Linear(x.shape[-1], self.num_classes)(x)
        return x
    
if __name__ == '__main__':
    model = Network(num_classes=18)

    img = torch.randn(3, 3,256,256)
    output = model(img)
    print(output.shape)

