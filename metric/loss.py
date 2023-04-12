import torch


def topk_loss(logit, y, criterion, k=5):
    '''
    criterion에 따른 k개의 top loss index를 반환합니다.
    '''
    pass
class loss_tracker():
    def __init__(self) -> None:
        self.total_loss = 0
        self.cnt = 0
        self.avg_loss = 0
    def update(self,loss,cnt):
        self.total_loss += loss
        self.cnt += cnt
        self.avg_loss = self.total_loss/self.cnt
    def get_loss(self):
        return self.avg_loss
    def reset(self):
        self.total_loss = 0
        self.cnt = 0
        self.avg_loss = 0

def classification_acc(logit,y):
    acc = torch.sum(logit.argmax(dim=1)==y)
    return acc