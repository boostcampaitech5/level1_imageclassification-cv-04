import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(196608, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        return x
    
if __name__ == '__main__':
    model = Network(num_classes=18)

    img = torch.randn(3, 3, 256, 256)
    output = model(img)
    print(output.shape)

