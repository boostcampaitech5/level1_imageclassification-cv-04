import torch
def CustomLoss(logit,label):
    class_loss = torch.nn.CrossEntropyLoss()
    age_loss = torch.nn.MSELoss()
    loss=class_loss(logit[:,:2],label[:,0])+\
    age_loss(logit[:,2:3],label[:,1:2])+\
    class_loss(logit[:,3:],label[:,2])
    return loss