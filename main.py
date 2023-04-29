import numpy as np
import os
import random
import wandb

import torch
import argparse
import logging
import json

from train import fit
from test import test
from datasets import create_dataloader
from datasets.dataset import CustomDataset, TestDataset
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
    model = __import__('models.model', fromlist='model').__dict__[args.model_name](args.num_classes, **args.model_param)
    _logger.info('# of params: {}'.format(np.sum([p.numel() for p in model.parameters()])))

    # load dataset
    trainset = CustomDataset(args=args, train=True)
    valset = CustomDataset(args=args, train=False)
    testset = TestDataset(args=args)
    
    # load dataloader
    trainloader = create_dataloader(dataset=trainset, batch_size=args.batch_size, shuffle=True)
    valloader = create_dataloader(dataset=valset, batch_size=args.batch_size, shuffle=False)
    testloader = create_dataloader(dataset=testset, batch_size=1, shuffle=False)

    # set criterion
    criterion = __import__('losses.loss', fromlist='loss').__dict__[args.loss](**args.loss_param)

    # set optimizer
    optimizer = __import__('torch.optim', fromlist='optim').__dict__[args.opt_name](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    if args.lr_scheduler:
        lr_scheduler = __import__('torch.optim.lr_scheduler', fromlist='lr_scheduler').__dict__[args.lr_scheduler](optimizer, **args.lr_scheduler_param)
    else:
        lr_scheduler = None

    # prepraring accelerator
    model, optimizer, trainloader, valloader, lr_scheduler = accelerator.prepare(
        model, optimizer, trainloader, valloader, lr_scheduler
    )

    # initialize wandb
    if args.use_wandb:
        wandb.init(name     = f'{args.exp_name}_{args.exp_num}.{args.user_name}', 
                   project  = args.project_name, 
                   entity   = args.entity,
                   config   = args)

    # fitting model
    fit(model        = model, 
        trainloader  = trainloader, 
        valloader    = valloader, 
        criterion    = criterion, 
        optimizer    = optimizer, 
        lr_scheduler = lr_scheduler,
        accelerator  = accelerator,
        savedir      = savedir,
        args         = args)

    # testing model
    test(model        = model,
         testloader   = testloader,
         accelerator  = accelerator,
         savedir      = savedir,
         args         = args)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Classification for Computer Vision")

    with open('config.json') as f:
        config = json.load(f)

    for key in config:
        parser_key = key.replace('_', '-')
        parser.add_argument(f'--{parser_key}', default=config[key], type=type(config[key]))

    args = parser.parse_args()

    run(args)