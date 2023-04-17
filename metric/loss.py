import torch

def CustomLoss(logit, label):
    class_loss = torch.nn.CrossEntropyLoss()
    age_loss = torch.nn.MSELoss()
    print(logit[0],label[0])
    print(logit[:5,2:3],label[:5,1:2])
    loss = [class_loss(logit[:,:2],label[:,0]),
            age_loss(logit[:,2:3].float(),label[:,1:2].float()),
            class_loss(logit[:,3:],label[:,2])]

    return loss