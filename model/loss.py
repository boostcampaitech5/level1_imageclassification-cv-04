import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()
        
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=0.25, size_average=True, device='cpu'):
#         super(FocalLoss, self).__init__()
#         """
#         gamma(int) : focusing parameter.
#         alpha(list) : alpha-balanced term.
#         size_average(bool) : whether to apply reduction to the output.
#         """
#         self.gamma = gamma
#         self.alpha = alpha
#         self.size_average = size_average
#         self.device = device


#     def forward(self, input, target):
#         # input : N * C (btach_size, num_class)
#         # target : N (batch_size)

#         CE = F.cross_entropy(input, target, reduction='none')  # -log(pt)
#         pt = torch.exp(-CE)  # pt
#         loss = (1 - pt) ** self.gamma * CE * self.alpha  # -(1-pt)^rlog(pt)

#         # if self.alpha is not None:
#         #     alpha = torch.Tensor(self.alpha).to(self.device)
#         #     # in case that a minority class is not selected when mini-batch sampling
#         #     if len(self.alpha) != len(torch.unique(target)):
#         #         temp = torch.zeros(len(self.alpha)).to(self.device)
#         #         temp[torch.unique(target)] = alpha.index_select(0, torch.unique(target))
#         #         alpha_t = temp.gather(0, target)
#         #         loss = alpha_t * loss
#         #     else:
#         #         alpha_t = alpha.gather(0, target)
#         #         loss = alpha_t * loss

#         if self.size_average:
#             loss = torch.mean(loss)

#         return loss


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

class CustomLoss(nn.Module):
    def __init__(self,w1:list=[1,1,1], w2:list=[1,1], w3:list=[1,1,1]):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mask_loss = nn.CrossEntropyLoss(weight=torch.Tensor(w1).to(self.device))
        self.gender_loss = nn.CrossEntropyLoss(weight=torch.Tensor(w2).to(self.device))
        self.age_loss = nn.CrossEntropyLoss(weight=torch.Tensor(w3).to(self.device))
    def forward(self, pred, label):
        label = label.long()
        mask_label = label//6
        gender_label = label%6//3
        age_label = label%3
        m_loss = self.mask_loss(pred[:,:3],mask_label.to(self.device))
        g_loss = self.gender_loss(pred[:,3:5],gender_label.to(self.device))
        a_loss = self.age_loss(pred[:,5:],age_label.to(self.device))
        return g_loss+a_loss+m_loss
    



