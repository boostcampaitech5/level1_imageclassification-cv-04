import torch
import torch.nn as nn
import torch.nn.functional as F

def confusion_matrix(model, data_iter, device, num_classes):
    with torch.no_grad():
        model.eval()
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
        for data, target in data_iter:
            model_pred, y_target = model(data.to(device)), target.to(device)
            _, y_pred = torch.max(model_pred.data, 1)
            for y_p, y_t in zip(y_pred, y_target):
                cm[y_t][y_p] += 1

    return cm

def confusion_matrix2(cm_data, num_classes):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for y_pred, y_target in cm_data:
        y_pred = torch.max(y_pred, 1)[1]
        for y_t, y_p in zip(y_target, y_pred):
            cm[y_t.item()][y_p.item()] += 1
    return cm


def accuracy(cm, num_classes):
    n_total, n_correct = torch.sum(cm), 0
    for i in range(num_classes):
        n_correct += cm[i][i]
    acc = (n_correct/n_total)

    return acc.item()


def precision(cm, num_classes):
    predict_positive = torch.sum(cm, 0) # TP + FP
    precision_val = torch.zeros(num_classes)
    for i in range(num_classes):
        val = cm[i][i] / predict_positive[i] # TP / (TP + FP)
        if not torch.isnan(val):
            precision_val[i] = val
    precision_val = torch.sum(precision_val).item() / num_classes

    return precision_val


def recall(cm, num_classes):
    actual_positive = torch.sum(cm, 1) # TP + FN
    recall_val = torch.zeros(num_classes)
    for i in range(num_classes):
        val = cm[i][i] / actual_positive[i] # TP / (TP + FN)
        if not torch.isnan(val):
            recall_val[i] = val
    recall_val = torch.sum(recall_val).item() / num_classes

    return recall_val


def f1_score(cm, num_classes):
    precision_val = precision(cm, num_classes)
    recall_val = recall(cm, num_classes)
    f1_score_val = 2 * ((precision_val * recall_val) / (precision_val + recall_val))

    return f1_score_val


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, device='cpu'):
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
        loss = (1 - pt) ** self.gamma * CE  # -(1-pt)^rlog(pt)

        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, dtype=torch.float).to(self.device)
            # in case that a minority class is not selected when mini-batch sampling
            if len(self.alpha) != len(torch.unique(target)):
                temp = torch.zeros(len(self.alpha)).to(self.device)
                temp[torch.unique(target)] = alpha.index_select(0, torch.unique(target))
                alpha_t = temp.gather(0, target)
                loss = alpha_t * loss
            else:
                alpha_t = alpha.gather(0, target)
                loss = alpha_t * loss

        if self.size_average:
            loss = torch.mean(loss)

        return loss
    

if __name__ == '__main__':
    num_classes = 3

    cm = [[5, 1, 1],
          [0, 5, 2],
          [1, 0, 5]]
    
    cm = torch.LongTensor(cm)
    print(accuracy(cm, num_classes))
    print(precision(cm, num_classes))
    print(recall(cm, num_classes))
    print(f1_score(cm, num_classes))



