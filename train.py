from dataloader import *
from model import *
from metric import *
from utils import *
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
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
from sklearn.model_selection import KFold, StratifiedKFold
import sys
import wandb_info
from model.model_finetune import fineTune
from model.loss import FocalLoss

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
        wandb.init(name = f'{args.wandb_exp_name}_{args.exp_num}_bs{args.batch_size}_ep{args.epochs}_{args.loss}_lr{args.learning_rate}_{args.load_model}.{args.user_name}',
                   project = args.wandb_project_name,
                   entity = args.wandb_entity,
                   config = args_dict)
        
    print(f'Seed\t>>\t{args.seed}')
    torch_seed(args.seed)

    device = accelerator.device
    print(f'The device is ready\t>>\t{device}')

    print('Make save_path')
    checkpoint_path = os.path.join(args.save_path, f'{args.wandb_exp_name}{args.exp_num}_bs{args.batch_size}_ep{args.epochs}_{optimizer_name}_lr{args.learning_rate}_{args.load_model}')
    os.makedirs(checkpoint_path, exist_ok=True)

    print(f'Transform\t>>\t{args.transform_list}')
    train_transform, config = get_transform(args)
    if args.use_wandb:
        wandb.config.update(config)

    val_transform = transforms.Compose([transforms.CenterCrop((300,300)),
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                             std=(0.229, 0.224, 0.225))])     

    train_set = KFoldDataset(csv_path = args.csv_path,
                             kfold = args.kfold,
                             transform=train_transform,
                             train=True)
    val_set = KFoldDataset(csv_path = args.csv_path,
                           kfold = args.kfold,
                           transform=val_transform,
                           train=False)           
    
    train_iter = DataLoader(train_set,
                            batch_size=args.batch_size,
                            num_workers=multiprocessing.cpu_count() // 2,
                            shuffle=True)
    val_iter = DataLoader(val_set,
                          batch_size=args.batch_size,
                          num_workers=multiprocessing.cpu_count() // 2,
                          shuffle=True)
        
    print('The model is ready ...')
    model = Classifier(args).to(device)
    
    if args.model_summary:
        print(summary(model, (3, 384, 384)))

    print('The optimizer is ready ...')
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print('The learning scheduler is ready ...')
    if args.lr_scheduler == 'steplr':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, verbose=True)
    elif args.lr_scheduler == 'reduce_lr_on_plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)

    print(f'The loss function({args.loss}) is ready ...')
    train_cnt = train_set.df['ans'].value_counts().sort_index()
    normedWeights = [1 - (x / sum(train_cnt)) for x in train_cnt]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    
    if args.loss == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "focalloss":
        # criterion = FocalLoss(alpha=0.1, device = device)
        criterion = FocalLoss()
    elif args.loss == 'labelsmooting':
        criterion = LabelSmoothingLoss(classes=args.num_classes, smoothing=args.labelsmoothing)
    elif args.loss == 'f1loss':
        criterion = F1Loss(classes=args.num_classes, epsilon=1e-7)

    #Accelerator 적용
    model, optimizer, train_iter, val_iter = accelerator.prepare(model, optimizer, train_iter, val_iter)
    
    print("Starting training ...")
    for epoch in range(args.epochs):
        start_time = time.time()
        train_epoch_loss = 0
        model.train()
        train_iter_loss=0
        for train_img, train_target in train_iter:
            optimizer.zero_grad()

            train_pred = model(train_img)
            train_iter_loss = criterion(train_pred, train_target)
            accelerator.backward(train_iter_loss)
            optimizer.step()

            train_epoch_loss += train_iter_loss

        train_loss = (train_epoch_loss / len(train_iter))

        train_cm = confusion_matrix(model, train_iter, device, args.num_classes)

        train_acc = accuracy(train_cm, args.num_classes)
        train_f1 = f1_score(train_cm, args.num_classes)

        # Validation
        with torch.no_grad():
            val_epoch_loss = 0
            val_iter_loss = 0
            model.eval()
            for val_img, val_target in val_iter:
                val_pred = model(val_img)
                val_iter_loss = criterion(val_pred, val_target).detach()

                val_epoch_loss += val_iter_loss
        val_loss = (val_epoch_loss / len(val_iter))

        val_cm = confusion_matrix(model, val_iter, device, args.num_classes)

        val_acc = accuracy(val_cm, args.num_classes)
        val_f1 = f1_score(val_cm, args.num_classes)

        if args.lr_scheduler:
            lr_scheduler.step(val_epoch_loss)

        print('time >> {:.4f}\tepoch >> {:04d}\ttrain_acc >> {:.4f}\ttrain_loss >> {:.4f}\ttrain_f1 >> {:.4f}\tval_acc >> {:.4f}\tval_loss >> {:.4f}\tval_f1 >> {:.4f}'
            .format(time.time()-start_time, epoch, train_acc, train_loss, train_f1, val_acc, val_loss, val_f1))
        
        if (epoch+1) % args.save_epoch == 0:
            # 모델의 parameter들을 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(checkpoint_path, f'epoch({epoch})_acc({val_acc:.3f})_loss({val_loss:.3f})_f1({val_f1:.3f})_state_dict.pt'))

        if args.use_wandb:
            wandb.log({'Train Acc': train_acc,
                    'Train Loss': train_loss,
                    'Train F1-Score': train_f1,
                    'Val Acc': val_acc,
                    'Val Loss': val_loss,
                    'Val F1-Score': val_f1})

        if (epoch+1) % args.save_epoch == 0:
            fig = plot_confusion_matrix(val_cm, args.num_classes, normalize=True, save_path=None)
            wandb.log({'Confusion Matrix': wandb.Image(fig, caption=f"Epoch-{epoch}")})

        if optimizer.param_groups[0]["lr"] < 1e-7:
            print(f'Early Stopping\t>>\tepoch : {epoch} lr : {optimizer.param_groups[0]["lr"]}')
            if (epoch+1) % args.save_epoch != 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(checkpoint_path, f'epoch({epoch})_acc({val_acc:.3f})_loss({val_loss:.3f})_f1({val_f1:.3f})_state_dict.pt'))
            break


if __name__ == '__main__':
    args_dict = {'seed' : 223,
                #  'csv_path' : '../input/data/train/train_info.csv',
                 'csv_path' : '../input/data/train/kfold4.csv',
                 'save_path' : './checkpoint',
                 'use_wandb' : True,
                 'wandb_exp_name' : 'kfold4_1_ce_reducelr',
                 'wandb_project_name' : 'Image_classification_mask',
                 'wandb_entity' : 'connect-cv-04',
                 'num_classes' : 18,
                 'model_summary' : False,
                 'batch_size' : 64,
                 'learning_rate' : 1e-4,
                 'epochs' : 100,
                #  'train_val_split': 0.8,
                #  'save_mode' : 'state_dict',
                 'save_epoch' : 10,
                 'load_model':'resnet50',
                 'loss' : "crossentropy",
                 'lr_scheduler' : 'reduce_lr_on_plateau', # default lr_scheduler = ''
                 'transform_path' : './transform_list.json',
                 'transform_list' : ['centercrop','resize', "randomrotation",'totensor', 'normalize'],
                #  'not_freeze_layer' : ['layer4'],
                 'weight_decay': 1e-2,
                 'labelsmoothing':0.1,
                 'kfold' : 1}
    wandb_data = wandb_info.get_wandb_info()
    args_dict.update(wandb_data)
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    # Config parser 하나만 넣어주면 됨(임시방편)
    run(args, args_dict)