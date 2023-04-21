from dataloader import *
from model import *
from metric import *
from utils import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms 
import multiprocessing
from tqdm import tqdm
import sys


def image_tensor_to_numpy(tensor_image):
  # If this is already a numpy image, just return it
  if type(tensor_image) == np.ndarray:
    return tensor_image
  
  # Make sure this is a tensor and not a variable
  if type(tensor_image) == Variable:
    tensor_image = tensor_image.data
  
  # Convert to numpy and move to CPU if necessary
  np_img = tensor_image.detach().cpu().numpy()
  
  # If there is no batch dimension, add one
  if len(np_img.shape) == 3:
    np_img = np_img[np.newaxis, ...]
  
  # Convert from BxCxHxW (PyTorch convention) to BxHxWxC (OpenCV/numpy convention)
  np_img = np_img.transpose(0, 2, 3, 1)
  
  return np_img


def normalize(tensor):
  x = tensor - tensor.min()
  x = x / (x.max() + 1e-9)
  return x


def grad_cam(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'The device is ready\t>>\t{device}')

    # Image size 조절과 tensor로만 만들어주면 됨(normalize까지는 해야 할 듯)
    transform = transforms.Compose([transforms.CenterCrop((300, 300)),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                         std=(0.229, 0.224, 0.225))])

    dataset = KFoldSplitDataset(csv_path = args.csv_path,
                           kfold = args.kfold,
                           transform=transform,
                           train=False,
                           split = args.split)    
     
    print(f'The number of testing images\t>>\t{len(dataset)}')

    dataiter = DataLoader(dataset,
                          batch_size=args.batch_size,
                          shuffle=True)

    print('Loading checkpoint ...')
    state_dict = torch.load(args.checkpoint)

    print('The model is ready ...')
    model = KFoldClassifier(args.num_classes, args.load_model)
    model.load_state_dict(state_dict['model_state_dict'])

    model.eval()

    save_feat = []
    def hook_feat(module, input, output):
        save_feat.append(output)
        return output

    save_grad = []
    def hook_grad(grad):
        """
        get a gradient from intermediate layers (dy / dA).
        See the .register-hook function for usage.
        :return grad: (Variable) gradient dy / dA
        """ 
        save_grad.append(grad)
        return grad

    # (1) Reister hook for storing layer activation of the target layer (bn5_2 in backbone)
    model.backbone.layer4[2].bn3.register_forward_hook(hook_feat)

    not_classify = torch.zeros(args.num_classes).to(device)
    classify = torch.zeros(args.num_classes).to(device)

    for img, target in tqdm(dataiter):
        # (2) Forward pass to hook features
        s = model(img)[0]

        # (3) Register hook for storing gradients
        save_feat[-1].register_hook(hook_grad)
        
        # (4) Backward score
        y = torch.argmax(s).item()

        if y == target:
           if classify[target] == 1:
              continue
           else:
              classify[target] += 1
        else:
           if not_classify[target] == 1:
              continue
           else:
              not_classify[target] += 1

        softmax_y = torch.nn.functional.softmax(s, dim=-1)[y]
        s_y = s[y]
        s_y.backward()

        # (5) Compute activation at global-average-pooling layer
        gap_layer  = torch.nn.AdaptiveAvgPool2d(1)
        alpha = gap_layer(save_grad[-1][0].squeeze())
        A = save_feat[-1].squeeze()

        # (6) Compute grad_CAM 
        relu_layer = torch.nn.ReLU()

        weighted_sum = torch.sum(alpha*A, dim=0)
        grad_CAM = relu_layer(weighted_sum)

        grad_CAM = grad_CAM.unsqueeze(0)
        grad_CAM = grad_CAM.unsqueeze(0)

        # (7) Upscale grad_CAM
        upscale_layer = torch.nn.Upsample(scale_factor=img.shape[-1]/grad_CAM.shape[-1], mode='bilinear')

        grad_CAM = upscale_layer(grad_CAM)
        grad_CAM = grad_CAM/torch.max(grad_CAM)

        # Plotting
        img_np = image_tensor_to_numpy(img)
        if len(img_np.shape) > 3:
          img_np = img_np[0]
        img_np = normalize(img_np)
        
        grad_CAM = grad_CAM.squeeze().detach().numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(img_np)
        plt.imshow(grad_CAM, cmap='jet', alpha = 0.5)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'GT : {target.item()} - Pred : {y}({softmax_y:.3f})')
        plt.savefig(os.path.join(args.save_path, f'grad_cam_{target.item()}_{y}.png'))
        plt.clf()

        if torch.sum(classify) == args.num_classes and torch.sum(not_classify) == args.num_classes:
           break



if __name__ == '__main__':
    args_dict = {'csv_path' : '../input/data/train/kfold4.csv',
                 'checkpoint' : './checkpoint/kfold4_2_cd_agedetection_reducelr71_bs64_ep100_adamw_lr0.0001_resnet50/epoch(29)_acc(0.798)_loss(0.536)_f1(0.716)_state_dict.pt',
                 'load_model':'resnet50',
                 'load_mode' : 'state_dict',
                 'num_classes' : 3,
                 'batch_size' : 1,
                 'model_summary' : False,
                 'save_path' : './grad_cam_age_swin',
                 'kfold' : 0,
                 'split' : 'age'}
    
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)
    
    # Config parser 하나만 넣어주면 됨(임시방편)
    grad_cam(args)