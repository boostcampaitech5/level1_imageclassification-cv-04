from dataloader import *
from model import *
from metric import *
from utils import *
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms 
from torch import optim
import numpy as np
import random
import os
import wandb
from torchsummary import summary
import time
import multiprocessing
import sklearn
import sys
import wandb_info
from accelerate import Accelerator

from tqdm import tqdm

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

def run(args, args_dict):
    
    accelerator = Accelerator(
        gradient_accumulation_steps = 1,
        mixed_precision             = 'fp16'
    )
    
    if args.weight_decay > 0:
        optimizer_name = 'adamw'
    else:
        optimizer_name = 'adam'

    if args.use_wandb:
        print('Initialize WandB ...')
        wandb.init(name = f'{args.wandb_exp_name}{args.exp_num}_bs{args.batch_size}_ep{args.epochs}_adam_lr{args.learning_rate}_{args.load_model}.{args.user_name}',
                   project = args.wandb_project_name,
                   entity = args.wandb_entity,
                   config = args_dict)
        
    print(f'Seed\t>>\t{args.seed}')
    torch_seed(args.seed)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = accelerator.device
    print(f'The device is ready\t>>\t{device}')

    print('Make save_path')
    checkpoint_path = os.path.join(args.save_path, f'{args.wandb_exp_name}_bs{args.batch_size}_ep{args.epochs}_{optimizer_name}_lr{args.learning_rate}_{args.load_model}')
    os.makedirs(checkpoint_path, exist_ok=True)

    print(f'Transform\t>>\t{args.transform_list}')
    transform, config = get_transform(args)
    if args.use_wandb:
        wandb.config.update(config)                                

    dataset = ClassificationDataset(csv_path = args.csv_path,
                                    transform=transform)

    n_train_set = int(args.train_val_split*len(dataset))
    train_set, val_set, train_idx, val_idx = train_valid_split_by_sklearn(dataset,args.train_val_split,args.seed)
    print(f'The number of training images\t>>\t{len(train_set)}')
    print(f'The number of validation images\t>>\t{len(val_set)}')



    print('The data loader is ready ...')
    train_sampler = weighted_sampler(dataset, train_idx, args.num_classes)
    val_sampler = weighted_sampler(dataset,val_idx, args.num_classes)
    
    train_iter = DataLoader(train_set,
                            batch_size=args.batch_size,
                            drop_last=True,
                            num_workers=multiprocessing.cpu_count() // 2
                            ,sampler = train_sampler
                            )   
    val_iter = DataLoader(val_set,
                          batch_size=args.batch_size,
                          drop_last=True,
                          num_workers=multiprocessing.cpu_count() // 2
                          ,sampler = val_sampler
                          )

    print('The model is ready ...')
    model = Classifier(args).to(device)
    if args.model_summary:
        print(summary(model, (3, 256, 256)))

    print('The optimizer is ready ...')
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print('The loss function is ready ...')
    criterion = nn.CrossEntropyLoss()
    
    #Accelerator 적용
    model, optimizer, train_iter, val_iter = accelerator.prepare(
        model, optimizer, train_iter, val_iter
    )

    print("Starting training ...")
    for epoch in range(args.epochs):
        start_time = time.time()
        train_epoch_loss = 0
        model.train()
        train_iter_loss=0
        pbar = tqdm(train_iter)
        for _,(train_img, train_target) in enumerate(pbar):
            pbar.set_description(f"Train. Epoch:{epoch}/{args.epochs} | Loss:{train_iter_loss:4.3f}")
            train_img, train_target = train_img.to(device), train_target.to(device)
            optimizer.zero_grad()

            train_pred = model(train_img)
            train_iter_loss = criterion(train_pred, train_target)
            # train_iter_loss.backward()
            accelerator.backward(train_iter_loss)
            optimizer.step()

            train_epoch_loss += train_iter_loss

        train_epoch_loss = train_epoch_loss / len(train_iter)

        train_cm = confusion_matrix(model, train_iter, device, args.num_classes)

        train_acc = accuracy(train_cm, args.num_classes)
        train_f1 = f1_score(train_cm, args.num_classes)

        # Validation
        with torch.no_grad():
            val_epoch_loss = 0
            val_iter_loss = 0
            model.eval()
            pbar = tqdm(val_iter)
            for _,(val_img, val_target) in enumerate(pbar):
                # val_img, val_target = val_img.to(device), val_target.to(device)
                pbar.set_description(f"Val. Epoch:{epoch}/{args.epochs} | Loss:{val_iter_loss:4.3f}")
                val_pred = model(val_img)
                val_iter_loss = criterion(val_pred, val_target).detach()

                val_epoch_loss += val_iter_loss
        val_epoch_loss = val_epoch_loss / len(val_iter)

        val_cm = confusion_matrix(model, val_iter, device, args.num_classes)

        val_acc = accuracy(val_cm, args.num_classes)
        val_f1 = f1_score(val_cm, args.num_classes)

        print('time >> {:.4f}\tepoch >> {:04d}\ttrain_acc >> {:.4f}\ttrain_loss >> {:.4f}\ttrain_f1 >> {:.4f}\tval_acc >> {:.4f}\tval_loss >> {:.4f}\tval_f1 >> {:.4f}'
              .format(time.time()-start_time, epoch, train_acc, train_epoch_loss, train_f1, val_acc, val_epoch_loss, val_f1))
        
        if (epoch+1) % args.save_epoch == 0:
            if args.save_mode == 'state_dict' or args.save_mode == 'both':
                # 모델의 parameter들을 저장
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(checkpoint_path, f'epoch({epoch})_acc({val_acc:.3f})_loss({val_epoch_loss:.3f})_f1({val_f1:.3f})_state_dict.pt'))
            if args.save_mode == 'model' or args.save_mode == 'both':
                # 모델 자체를 저장
                torch.save(model, os.path.join(checkpoint_path, f'epoch({epoch})_acc({val_acc:.3f})_loss({val_epoch_loss:.3f})_f1({val_f1:.3f})_model.pt'))

        if args.use_wandb:
            wandb.log({'Train Acc': train_acc,
                       'Train Loss': train_epoch_loss,
                       'Train F1-Score': train_f1,
                       'Val Acc': val_acc,
                       'Val Loss': val_epoch_loss,
                       'Val F1-Score': val_f1})

            if (epoch+1) % args.save_epoch == 0:
                fig = plot_confusion_matrix(val_cm, args.num_classes, normalize=True, save_path=None)
                wandb.log({'Confusion Matrix': wandb.Image(fig, caption=f"Epoch-{epoch}")})
                # wandb.log({"Confusion Matrix Plot" : wandb.plot.confusion_matrix(probs=None,
                #                                                             preds=pred_list, y_true=target_list,
                #                                                             class_names=list(map(str,range(0, 18))))})
                # # WARNING wandb.plots.* functions are deprecated and will be removed in a future release. Please use wandb.plot.* instead.
                # wandb.log({'Confusion Matrix Heatmap': wandb.plots.HeatMap(list(range(0,18)), list(range(0,18)), val_cm, show_text=True)})


if __name__ == '__main__':
    args_dict = {'seed' : 223,
                 'csv_path' : './input/data/train/train_info.csv',
                 'save_path' : './checkpoint',
                 'use_wandb' : False,
                 'wandb_exp_name' : 'exp5',
                 'wandb_project_name' : 'Transform_Exp',
                 'wandb_entity' : 'connect-cv-04',
                 'num_classes' : 18,
                 'model_summary' : True,
                 'batch_size' : 64,
                 'learning_rate' : 1e-4,
                 'epochs' : 100,
                 'train_val_split': 0.8,
                 'save_mode' : 'model',
                 'save_epoch' : 10,
                 'load_model':'resnet50',
                 'transform_path' : './transform_list.json',
                 'transform_list' : ['resize', 'randomhorizontalflip', 'randomrotation', 'totensor', 'normalize'],
                 'not_freeze_layer' : ['layer4'],
                 'weight_decay': 1e-2}
    wandb_data = wandb_info.get_wandb_info()
    args_dict.update(wandb_data)
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    # Config parser 하나만 넣어주면 됨(임시방편)
    run(args, args_dict)
    
    
