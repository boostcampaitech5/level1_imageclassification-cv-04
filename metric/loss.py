def topk_loss(logit, y, criterion, k=5):
    '''
    criterion에 따른 k개의 top loss index를 반환합니다.
    '''
    