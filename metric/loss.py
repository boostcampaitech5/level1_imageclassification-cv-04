import torch

def CustomLoss(logit, label):
    mask_loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1,5,3]).cuda())
    gender_loss = torch.nn.CrossEntropyLoss()
    age_loss = torch.nn.MSELoss()
    #age_loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1,3,5]).cuda())
#     loss = [gender_loss(logit[:,:2],label[:,0]),
#             age_loss(logit[:,2:5],label[:,1]),
#             mask_loss(logit[:,5:],label[:,2])]
    loss = [gender_loss(logit[:,:2],label[:,0]),
            age_loss(logit[:,2:3].float(),label[:,1:2].float()),
            mask_loss(logit[:,3:],label[:,2])]

    return loss