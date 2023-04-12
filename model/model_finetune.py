
import torch
import torch.nn as nn

# crop 한 이미지 사이즈와 output 클래스 개수 
def fineTune(model,model_name,img_size,class_num):
    if model_name == "Resnet18":
        model.fc = nn.Linear(model.fc.in_features,class_num)
        
        return model
    