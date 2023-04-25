import os 
import torch
import numpy as np
import random

def chechpoint_init(args):
    print('Make save_path')
    checkpoint_path = os.path.join(args.save_path, f'{args.wandb_exp_name}{args.exp_num}\
                                   _bs{args.batch_size}_ep{args.epochs}_{args.opt_name}_lr{args.learning_rate}_{args.load_model}')
    os.makedirs(checkpoint_path, exist_ok=True)
    return checkpoint_path

class metric_tracker():
    def __init__(self):
        self.cm_data = []
        self.loss = 0
        self.acc = 0
        self.cnt = 0
    
    def update_loss(self,loss,cnt):
        self.loss += loss
        self.cnt += cnt
    def update_cm_data(self,cm_data):
        self.cm_data.append(cm_data)

    def get_loss(self):
        return self.loss/self.cnt
    def get_cm_data(self):
        return self.cm
    
    def reset(self):
        self.cm = []
        self.loss = 0
        self.acc = 0
        self.cnt = 0

def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU 
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)