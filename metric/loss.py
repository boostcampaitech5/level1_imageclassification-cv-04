from torch import nn


def topk_loss(logit, y, criterion, k=5):
    '''
    criterion에 따른 k개의 top loss index를 반환합니다.
    '''
    pass
def set_loss(target_loss):
    loss_list = {"BCELoss":nn.BCELoss, "CrossEntropyLoss":nn.CrossEntropyLoss,
                 "MSELoss":nn.MSELoss}
    return loss_list[target_loss]()