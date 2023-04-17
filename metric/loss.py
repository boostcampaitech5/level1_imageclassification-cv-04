import torch

def CustomLoss(logit, label):
    class_loss = torch.nn.CrossEntropyLoss()
    age_loss = torch.nn.MSELoss()

    loss = [class_loss(logit[:,:2],label[:,0]),
            class_loss(logit[:,2:5],label[:,1]),
            class_loss(logit[:,5:],label[:,2])]

    return loss