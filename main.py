import numpy as np
import os
import random
import wandb

import torch
import argparse
import timm
import logging

from train import fit
from models import *
from datasets import create_dataset, create_dataloader
from log import setup_default_logging

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

    setup_default_logging(log_path=os.path.join(savedir,'log.txt'))
    torch_seed(args.seed)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # build Model
    model = ResNet18(num_classes=args.num_classes)
    model.to(device)
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # load dataset
    trainset, testset = create_dataset(datadir=args.datadir, dataname=args.dataname, aug_name=args.aug_name)
    
    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    testloader = create_dataloader(dataset=testset, batch_size=256, shuffle=False)

    # set training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.opt_name](model.parameters(), lr=args.lr)

    # scheduler
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    # initialize wandb
    wandb.init(name=args.exp_name, project='DSBA-study', config=args)

    # fitting model
    fit(model        = model, 
        trainloader  = trainloader, 
        testloader   = testloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        scheduler    = scheduler,
        epochs       = args.epochs, 
        savedir      = savedir,
        log_interval = args.log_interval,
        device       = device)

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