import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

        
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True, device='cpu'):
        super(FocalLoss, self).__init__()
        """
        gamma(int) : focusing parameter.
        alpha(list) : alpha-balanced term.
        size_average(bool) : whether to apply reduction to the output.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.device = device


    def forward(self, input, target):
        # input : N * C (btach_size, num_class)
        # target : N (batch_size)

        CE = F.cross_entropy(input, target, reduction='none')  # -log(pt)
        pt = torch.exp(-CE)  # pt
        loss = (1 - pt) ** self.gamma * CE * self.alpha  # -(1-pt)^rlog(pt)

    
        if self.size_average:
            loss = torch.mean(loss)

        return loss