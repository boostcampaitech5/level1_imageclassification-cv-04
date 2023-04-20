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
from model.loss import FocalLoss
import logging

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
    
    accelerator_mask = Accelerator(
        gradient_accumulation_steps = 1,
        mixed_precision             = 'fp16'
    )
    accelerator_gender = Accelerator(
        gradient_accumulation_steps = 1,
        mixed_precision             = 'fp16'
    )
    accelerator_age = Accelerator(
        gradient_accumulation_steps = 1,
        mixed_precision             = 'fp16'
    )
    
    if args.weight_decay > 0:
        optimizer_name = 'adamw'
    else:
        optimizer_name = 'adam'

    if args.use_wandb:
        print('Initialize WandB ...')
        wandb.init(name = f'{args.wandb_exp_name}_{args.exp_num}_bs{args.batch_size}_ep{args.epochs}_{optimizer_name}_lr{args.learning_rate}_{args.load_model}.{args.user_name}',
                   project = args.wandb_project_name,
                   entity = args.wandb_entity,
                   config = args_dict)
        
    print(f'Seed\t>>\t{args.seed}')
    torch_seed(args.seed)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_mask = accelerator_mask.device
    device_gender = accelerator_gender.device
    device_age = accelerator_age.device
    print(f'The device_mask is ready\t>>\t{device_mask}')
    print(f'The device_gender is ready\t>>\t{device_gender}')
    print(f'The device_age is ready\t>>\t{device_age}')

    print('Make save_path')
    checkpoint_path = os.path.join(args.save_path, 
                                   f'{args.wandb_exp_name}_{args.exp_num}_bs{args.batch_size}_ep{args.epochs}_{optimizer_name}_lr{args.learning_rate}_{args.load_model}')
    os.makedirs(checkpoint_path, exist_ok=True)

    print(f'Transform\t>>\t{args.transform_list}')
    transform, config = get_transform(args)
    if args.use_wandb:
        wandb.config.update(config)                                

    # dataset = ClassificationDataset(csv_path = args.csv_path,
    #                                 transform=transform)
    train_set = KFoldDataset(csv_path = '../input/data/train/kfold4.csv',
                                transform=transform,
                                kfold=0, train=True)
    val_set = KFoldDataset(csv_path = '../input/data/train/kfold4.csv',
                                transform=transform,
                                kfold=0, train=False)

    #n_train_set = int(args.train_val_split*len(dataset))
    # train_set, val_set, train_idx, val_idx = train_valid_split_by_sklearn(dataset,args.train_val_split,args.seed)
    print(f'The number of training images\t>>\t{len(train_set)}')
    print(f'The number of validation images\t>>\t{len(val_set)}')

    print('The data loader is ready ...')
    #train_sampler = weighted_sampler(dataset, train_idx, args.num_classes)
    #val_sampler = weighted_sampler(dataset,val_idx, args.num_classes)
    
    train_iter = DataLoader(train_set,
                            batch_size=args.batch_size,
                            num_workers=multiprocessing.cpu_count() // 2,
                            shuffle=True#sampler = train_sampler
                            )   
    val_iter = DataLoader(val_set,
                          batch_size=args.batch_size,
                          num_workers=multiprocessing.cpu_count() // 2,
                          shuffle=True#sampler = val_sampler
                          )

    print('The model is ready ...')
    model_mask = Classifier2(args.load_model, args.num_mask_classes).to(device_mask)
    model_gender = Classifier2(args.load_model, args.num_gender_classes).to(device_gender)
    # model_age = Classifier2(args.load_model, args.num_age_classes).to(device_gender)
    model_age = AgeClassifier('resnet50', 1).to(device_age)

    if args.model_summary:
        print('model_mask')
        print(summary(model_mask, (3, 224, 224)))
        print('model_gender')
        print(summary(model_gender, (3, 224, 224)))
        print('model_age')
        print(summary(model_age, (3, 224, 224)))

    print('The optimizer is ready ...')
    optimizer_mask = optim.Adam(params=model_mask.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_gender = optim.Adam(params=model_gender.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_age = optim.Adam(params=model_age.parameters(), lr=1e-3, weight_decay=args.weight_decay)

    print('The loss function is ready ...')
    
    # train_cnt = dataset.df['ans'][train_idx].value_counts().sort_index()
    # normedWeights = [1 - (x / sum(train_cnt)) for x in train_cnt]
    
    # df_mask = dataset.df['ans'][train_idx] // 6
    # df_gender = (dataset.df['ans'][train_idx] // 3) % 2
    # df_age = dataset.df['ans'][train_idx] % 3

    # train_mask_cnt = df_mask.value_counts().sort_index()
    # train_gender_cnt = df_gender.value_counts().sort_index()
    # train_age_cnt = df_age.value_counts().sort_index()

    # normedWeights_mask = [1 - (x / sum(train_mask_cnt)) for x in train_mask_cnt]
    # normedWeights_gender = [1 - (x / sum(train_gender_cnt)) for x in train_gender_cnt]
    # normedWeights_age = [1 - (x / sum(train_age_cnt)) for x in train_age_cnt]

    # normedWeights_mask = torch.FloatTensor(normedWeights_mask).to(device_mask)
    # normedWeights_gender = torch.FloatTensor(normedWeights_gender).to(device_gender)
    # normedWeights_age = torch.FloatTensor(normedWeights_age).to(device_age)

    criterion = nn.CrossEntropyLoss()
    criterion_age = nn.MSELoss()
    # if args.loss == "crossentropy":
    #     criterion_mask = nn.CrossEntropyLoss(normedWeights_mask)
    #     criterion_gender = nn.CrossEntropyLoss(normedWeights_gender)
    #     criterion_age = nn.CrossEntropyLoss(normedWeights_age)
    # elif args.loss == "focalloss":
    #     criterion_mask = FocalLoss(alpha=0.1, device = device_mask)
    #     criterion_gender = FocalLoss(alpha=0.1, device = device_gender)
    #     criterion_age = FocalLoss(alpha=0.1, device = device_age)
    
    #Accelerator 적용
    model_mask, optimizer_mask, train_iter, val_iter = accelerator_mask.prepare(model_mask, optimizer_mask, train_iter, val_iter)
    model_gender, optimizer_gender, train_iter, val_iter = accelerator_gender.prepare(model_gender, optimizer_gender, train_iter, val_iter)
    model_age, optimizer_age, train_iter, val_iter = accelerator_age.prepare(model_age, optimizer_age, train_iter, val_iter)

    best_val_f1 = 0 # for saving the best score along epochs
    logger = logging.getLogger('train')
    handler = logging.FileHandler(checkpoint_path + '/model.log')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for k, v in zip(args._fields, args):
        logger.info(k + ' = ' + str(v))

    print("Starting training ...")
    for epoch in range(args.epochs):
        start_time = time.time()

        train_mask_epoch_loss = 0
        train_gender_epoch_loss = 0
        train_age_epoch_loss = 0
        train_sum_epoch_loss = 0

        model_mask.train()
        model_gender.train()
        model_age.train()

        train_mask_iter_loss=0
        train_gender_iter_loss=0
        train_age_iter_loss=0
        train_sum_iter_loss=0

        train_mask_cm_data = []
        train_gender_cm_data = []
        train_age_cm_data = []
        train_cm_data = []

        pbar_train = tqdm(train_iter)
        for _,(train_img, train_target, train_targetage) in enumerate(pbar_train):
            pbar_train.set_description(f"Train. Epoch:{epoch}/{args.epochs} | Loss:{train_sum_iter_loss:4.3f}")
            
            train_mask_target = train_target // 6           # 'Wear': 0, 'Incorrect': 1, 'Not Wear': 2
            train_gender_target = (train_target // 3) % 2   # 'Male': 0, 'Female': 1
            # train_age_target = train_target % 3             # '< 30': 0, '>= 30 and < 60': 1, '>= 60': 2
            train_age_target = train_targetage if train_targetage < 60 else train_targetage + 15

            train_img, train_mask_target = train_img.to(device_mask), train_mask_target.to(device_mask)
            train_img, train_gender_target = train_img.to(device_gender), train_gender_target.to(device_gender)
            train_img, train_age_target = train_img.to(device_age), train_age_target.to(device_age)

            optimizer_mask.zero_grad()
            optimizer_gender.zero_grad()
            optimizer_age.zero_grad()

            train_mask_pred = model_mask(train_img)
            train_gender_pred = model_gender(train_img)
            train_age_pred = model_age(train_img)
            # train_pred = torch.max(train_mask_pred, 1)[1] * 6 + torch.max(train_gender_pred, 1)[1] * 3 + torch.max(train_age_pred, 1)[1]
            train_pred = torch.max(train_mask_pred, 1)[1] * 6 + torch.max(train_gender_pred, 1)[1] * 3 + (train_age_pred // 30)

            train_mask_cm_data.append([torch.max(train_mask_pred, 1)[1], train_mask_target])
            train_gender_cm_data.append([torch.max(train_gender_pred, 1)[1], train_gender_target])
            # train_age_cm_data.append([torch.max(train_age_pred, 1)[1], train_age_target])
            train_age_cm_data.append([train_age_pred // 30, train_age_target // 30])
            train_cm_data.append([train_pred, train_target])

            # train_mask_iter_loss = criterion_mask(train_mask_pred, train_mask_target)
            # train_gender_iter_loss = criterion_gender(train_gender_pred, train_gender_target)
            # train_age_iter_loss = criterion_age(train_age_pred, train_age_target)
            train_mask_iter_loss = criterion(train_mask_pred, train_mask_target)
            train_gender_iter_loss = criterion(train_gender_pred, train_gender_target)
            # train_age_iter_loss = criterion(train_age_pred, train_age_target)
            train_age_iter_loss = criterion_age(train_age_pred, train_age_target)
            train_sum_iter_loss = train_age_iter_loss + train_gender_iter_loss + train_age_iter_loss # not for calculation but just for report

            # train_iter_loss.backward()
            accelerator_mask.backward(train_mask_iter_loss)
            accelerator_gender.backward(train_gender_iter_loss)
            accelerator_age.backward(train_age_iter_loss)

            optimizer_mask.step()
            optimizer_gender.step()
            optimizer_age.step()

            train_mask_epoch_loss += train_mask_iter_loss
            train_gender_epoch_loss += train_gender_iter_loss
            train_age_epoch_loss += train_age_iter_loss
            train_sum_epoch_loss += train_sum_iter_loss # not for calculation but just for report

        pbar_train.close()

        train_sum_epoch_loss = train_sum_epoch_loss / len(train_iter)

        train_mask_cm = confusion_matrix2(train_mask_cm_data, args.num_mask_classes)
        train_gender_cm = confusion_matrix2(train_gender_cm_data, args.num_gender_classes)
        train_age_cm = confusion_matrix2(train_age_cm_data, args.num_age_classes)
        train_cm = confusion_matrix2(train_cm_data, args.num_classes)

        train_mask_acc = accuracy(train_mask_cm, args.num_mask_classes)
        train_gender_acc = accuracy(train_gender_cm, args.num_gender_classes)
        train_age_acc = accuracy(train_age_cm, args.num_age_classes)
        train_acc = accuracy(train_cm, args.num_classes)

        train_mask_f1 = f1_score(train_mask_cm, args.num_mask_classes)
        train_gender_f1 = f1_score(train_gender_cm, args.num_gender_classes)
        train_age_f1 = f1_score(train_age_cm, args.num_age_classes)
        train_f1 = f1_score(train_cm, args.num_classes)

        # Validation
        with torch.no_grad():
            val_mask_epoch_loss = 0
            val_gender_epoch_loss = 0
            val_age_epoch_loss = 0
            val_sum_epoch_loss = 0

            val_mask_iter_loss = 0
            val_gender_iter_loss = 0
            val_age_iter_loss = 0            
            val_sum_iter_loss = 0

            model_mask.eval()
            model_gender.eval()
            model_age.eval()

            val_mask_cm_data = []
            val_gender_cm_data = []
            val_age_cm_data = []
            val_cm_data = []

            pbar_val = tqdm(val_iter)
            for _,(val_img, val_target, val_targetage) in enumerate(pbar_val):
                # val_img, val_target = val_img.to(device), val_target.to(device)
                pbar_val.set_description(f"Val. Epoch:{epoch}/{args.epochs} | Loss:{val_sum_iter_loss:4.3f}")

                val_mask_target = val_target // 6           # 'Wear': 0, 'Incorrect': 1, 'Not Wear': 2
                val_gender_target = (val_target // 3) % 2   # 'Male': 0, 'Female': 1
                # val_age_target = val_target % 3             # '< 30': 0, '>= 30 and < 60': 1, '>= 60': 2
                val_age_target = val_targetage if val_targetage < 60 else val_targetage + 15

                val_mask_pred = model_mask(val_img)
                val_gender_pred = model_gender(val_img)
                val_age_pred = model_age(val_img)
                # val_pred = torch.max(val_mask_pred, 1)[1] * 6 + torch.max(val_gender_pred, 1)[1] * 3 + torch.max(val_age_pred, 1)[1]
                val_pred = torch.max(val_mask_pred, 1)[1] * 6 + torch.max(val_gender_pred, 1)[1] * 3 + (val_age_pred // 30)
                
                val_mask_cm_data.append([torch.max(val_mask_pred, 1)[1], val_mask_target])
                val_gender_cm_data.append([torch.max(val_gender_pred, 1)[1], val_gender_target])
                # val_age_cm_data.append([torch.max(val_age_pred, 1)[1], val_age_target])
                val_age_cm_data.append([val_age_pred // 30, val_age_target // 30])
                val_cm_data.append([val_pred, val_target])

                # val_mask_iter_loss = criterion_mask(val_mask_pred, val_mask_target).detach()
                # val_gender_iter_loss = criterion_gender(val_gender_pred, val_gender_target).detach()
                # val_age_iter_loss = criterion_age(val_age_pred, val_age_target).detach()
                val_mask_iter_loss = criterion(val_mask_pred, val_mask_target).detach()
                val_gender_iter_loss = criterion(val_gender_pred, val_gender_target).detach()
                # val_age_iter_loss = criterion(val_age_pred, val_age_target).detach()
                val_age_iter_loss = criterion_age(val_age_pred, val_age_target).detach()
                val_sum_iter_loss = val_age_iter_loss + val_gender_iter_loss + val_age_iter_loss # not for calculation but just for report

                val_mask_epoch_loss += val_mask_iter_loss
                val_gender_epoch_loss += val_gender_iter_loss
                val_age_epoch_loss += val_age_iter_loss
                val_sum_epoch_loss += val_sum_iter_loss # not for calculation but just for report

            pbar_val.close()

        val_sum_epoch_loss = val_sum_epoch_loss / len(val_iter)

        val_mask_cm = confusion_matrix2(val_mask_cm_data, args.num_mask_classes)
        val_gender_cm = confusion_matrix2(val_gender_cm_data, args.num_gender_classes)
        val_age_cm = confusion_matrix2(val_age_cm_data, args.num_age_classes)
        val_cm = confusion_matrix2(val_cm_data, args.num_classes)

        val_mask_acc = accuracy(val_mask_cm, args.num_mask_classes)
        val_gender_acc = accuracy(val_gender_cm, args.num_gender_classes)
        val_age_acc = accuracy(val_age_cm, args.num_age_classes)        
        val_acc = accuracy(val_cm, args.num_classes)
        
        val_mask_f1 = f1_score(val_mask_cm, args.num_mask_classes)
        val_gender_f1 = f1_score(val_gender_cm, args.num_gender_classes)
        val_age_f1 = f1_score(val_age_cm, args.num_age_classes)
        val_f1 = f1_score(val_cm, args.num_classes)

        print('time >> {:.4f}\tepoch >> {:04d}\ttrain_acc >> {:.4f}\ttrain_loss >> {:.4f}\ttrain_f1 >> {:.4f}\tval_acc >> {:.4f}\tval_loss >> {:.4f}\tval_f1 >> {:.4f}'
              .format(time.time()-start_time, epoch, train_acc, train_sum_epoch_loss, train_f1, val_acc, val_sum_epoch_loss, val_f1))
        
        if best_val_f1 <= val_f1:
            if args.save_mode == 'state_dict' or args.save_mode == 'both':
                # 모델의 parameter들을 저장
                torch.save({
                    'epoch': epoch,
                    'model_mask_state_dict': model_mask.state_dict(),
                    'model_gender_state_dict': model_gender.state_dict(),
                    'model_age_state_dict': model_age.state_dict(),
                    'optimizer_mask_state_dict': optimizer_mask.state_dict(),
                    'optimizer_gender_state_dict': optimizer_gender.state_dict(),
                    'optimizer_age_state_dict': optimizer_age.state_dict(),
                    }, os.path.join(checkpoint_path, f'best_state_dict.pt'))
            if args.save_mode == 'model' or args.save_mode == 'both':
                # 모델 자체를 저장
                torch.save(model, os.path.join(checkpoint_path, f'best_model.pt'))
            logger.info(f'epoch({epoch}), train_acc({train_acc:.3f}), train_loss({train_sum_epoch_loss:.3f}), train_f1({train_f1:.3f}), val_acc({val_acc:.3f}), val_loss({val_sum_epoch_loss:.3f}), val_f1({val_f1:.3f})')

        if args.use_wandb:
            wandb.log({'Train Acc': train_acc,
                       'Train mask Acc': train_mask_acc,
                       'Train gender Acc': train_gender_acc,
                       'Train age Acc': train_age_acc,
                       'Train Loss': train_sum_epoch_loss,
                       'Train mask Loss': train_mask_epoch_loss,
                       'Train gender Loss': train_gender_epoch_loss,
                       'Train age Loss': train_age_epoch_loss,
                       'Train F1-Score': train_f1,
                       'Train mask F1-Score': train_mask_f1,
                       'Train gender F1-Score': train_gender_f1,
                       'Train age F1-Score': train_age_f1,
                       'Val Acc': val_acc,
                       'Val mask Acc': val_mask_acc,
                       'Val gender Acc': val_gender_acc,
                       'Val age Acc': val_age_acc,
                       'Val Loss': val_sum_epoch_loss,
                       'Val mask Loss': val_mask_epoch_loss,
                       'Val gender Loss': val_gender_epoch_loss,
                       'Val age Loss': val_age_epoch_loss,
                       'Val F1-Score': val_f1,
                       'Val mask F1-Score': val_mask_f1,
                       'Val gender F1-Score': val_gender_f1,
                       'Val age F1-Score': val_age_f1}, step=epoch)

            if best_val_f1 <= val_f1:
                fig = plot_confusion_matrix(val_cm, args.num_classes, normalize=True, save_path=None)
                wandb.log({'Confusion Matrix': wandb.Image(fig, caption=f"Epoch-{epoch}")}, step=epoch)
                fig_mask = plot_confusion_matrix(val_mask_cm, args.num_mask_classes, normalize=True, save_path=None)
                wandb.log({'Confusion Matrix (mask)': wandb.Image(fig_mask, caption=f"Epoch-{epoch}")}, step=epoch)
                fig_gender = plot_confusion_matrix(val_gender_cm, args.num_gender_classes, normalize=True, save_path=None)
                wandb.log({'Confusion Matrix (gender)': wandb.Image(fig_gender, caption=f"Epoch-{epoch}")}, step=epoch)
                fig_age = plot_confusion_matrix(val_age_cm, args.num_age_classes, normalize=True, save_path=None)
                wandb.log({'Confusion Matrix (age)': wandb.Image(fig_age, caption=f"Epoch-{epoch}")}, step=epoch)
                # wandb.log({"Confusion Matrix Plot" : wandb.plot.confusion_matrix(probs=None,
                #                                                             preds=pred_list, y_true=target_list,
                #                                                             class_names=list(map(str,range(0, 18))))})
                # # WARNING wandb.plots.* functions are deprecated and will be removed in a future release. Please use wandb.plot.* instead.
                # wandb.log({'Confusion Matrix Heatmap': wandb.plots.HeatMap(list(range(0,18)), list(range(0,18)), val_cm, show_text=True)})
        if best_val_f1 <= val_f1:
            best_val_f1 = val_f1

if __name__ == '__main__':
    args_dict = {'seed' : 223,
                 'csv_path' : '../input/data/train/train_info.csv',
                 'save_path' : './checkpoint',
                 'num_classes' : 18,
                 'num_mask_classes' : 3,
                 'num_gender_classes' : 2,
                 'num_age_classes' : 3,
                 'model_summary' : False,
                 'batch_size' : 32,#64,#16,
                 'learning_rate' : 5e-6,
                 'epochs' : 100,
                 'train_val_split': 0.8,
                 'save_mode' : 'state_dict', #'model'
                 'save_epoch' : 10,
                 'load_model': 'vit_small_patch16_224', #'densenet121', #'resnet18',
                 'loss' : "crossentropy",
                 'transform_path' : './utils/transform_list.json',
                 'transform_list' : ['resize', 'randomhorizontalflip', 'randomrotation', 'totensor', 'normalize'],#['resize', 'randomhorizontalflip', 'randomrotation', 'totensor', 'normalize'],
                 'not_freeze_layer' : ['layer4'],
                 'weight_decay': 5e-4}
    wandb_data = wandb_info.get_wandb_info()
    args_dict.update(wandb_data)
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    # Config parser 하나만 넣어주면 됨(임시방편)
    run(args, args_dict)
    
