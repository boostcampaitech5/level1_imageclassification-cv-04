import numpy as np
import os
import random
import wandb

import torch
import argparse
import timm
import logging

from train import fit
# from models import * # timm을 main.py에서 바로 사용하면 삭제
from datasets import create_dataset, create_dataloader
from log import setup_default_logging

from accelerate import Accelerator

_logger = logging.getLogger('train')

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


def run(args):
    # make save directory
    savedir = os.path.join(args.savedir, args.exp_name)
    os.makedirs(savedir, exist_ok=True)

    # set logger
    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))

    # set seed
    torch_seed(args.seed)

    # set accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = args.grad_accum_steps,
        mixed_precision             = args.mixed_precision
    )
    _logger.info('Device: {}'.format(accelerator.device))

    # build Model
    model = create_model(args.backbone, pretrained=True, num_classes=args.num_classes) # create_model 작성 필요
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # load dataset
    trainset, testset = create_dataset(datadir=args.datadir, dataname=args.dataname, aug_list=args.aug_list) # create_dataset 변경 필요
    
    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    testloader = create_dataloader(dataset=testset, batch_size=args.batch_size, shuffle=False)

    # set criterion
    criterion = __import__('Folder Name').__dict__[args.loss](**args.loss_param) # Loss 선택할 수 있도록 작성 필요

    # set optimizer
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.opt_name](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    if args.lr_scheduler:
        lr_scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[args.lr_scheduler](optimizer, **args.lr_sheduler_param)
    else:
        lr_scheduler = None

    # prepraring accelerator
    model, optimizer, trainloader, testloader, lr_scheduler = accelerator.prepare(
        model, optimizer, trainloader, testloader, lr_scheduler
    )

    # load checkpoints
    if args.ckpdir:
        accelerator.load_state(args.ckpdir)

    # initialize wandb
    if args.use_wandb:
        wandb.init(name     = f'{args.exp_name}_{args.exp_num}.{args.user_name}', 
                   project  = args.project_name, 
                   entity   = args.entity,
                   config   = args)

    # fitting model
    fit(model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        lr_scheduler = lr_scheduler,
        accelerator  = accelerator,
        epochs       = args.epochs, 
        savedir      = savedir,
        log_interval = args.log_interval,
        use_wandb    = args.use_wandb)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Classification for Computer Vision")
    # exp setting
    parser.add_argument('--exp-name',type=str,help='experiment name')
    parser.add_argument('--datadir',type=str,default='/data',help='data directory')
    parser.add_argument('--savedir',type=str,default='./saved_model',help='saved model directory')

    # datasets
    parser.add_argument('--dataname',type=str,default='CIFAR100',choices=['CIFAR10','CIFAR100'],help='target dataname')
    parser.add_argument('--num-classes',type=int,default=100,help='target classes')

    # optimizer
    parser.add_argument('--opt-name',type=str,choices=['SGD','Adam'],help='optimizer name')
    parser.add_argument('--lr',type=float,default=0.1,help='learning_rate')

    # scheduler
    parser.add_argument('--use_scheduler',action='store_true',help='use sheduler')

    # augmentation
    parser.add_argument('--aug-name',type=str,choices=['default','weak','strong'],help='augmentation type')

    # train
    parser.add_argument('--epochs',type=int,default=50,help='the number of epochs')
    parser.add_argument('--batch-size',type=int,default=128,help='batch size')
    parser.add_argument('--log-interval',type=int,default=10,help='log interval')

    # seed
    parser.add_argument('--seed',type=int,default=223,help='223 is my birthday')

    args = parser.parse_args()

    run(args)