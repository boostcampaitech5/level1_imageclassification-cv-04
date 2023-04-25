
#for model train
from dataloader import *
from model import *
from metric import *

from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms 
from torch import optim
import multiprocessing
from accelerate import Accelerator

#array utils
import numpy as np

#other utils
import random
import os
import wandb
import time
from tqdm import tqdm
import argparse

#user made utils
from utils.config import load_config
from utils import plot, sampler, transform, config, util_tool

#for train log
from torchsummary import summary
import logging
from logger import wandb_util

from model.model_finetune import fineTune
from model.loss import FocalLoss


#추후 util로 옮길 함수
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

def run(args, config):
    
    accelerator = Accelerator(
        gradient_accumulation_steps = args.grad_accum_param,
        mixed_precision = args.mixed_precision
    )
    
    optimizer_name = args.opt_name

    #wandb initialize
    if args.use_wandb:
        wandb_util.wandb_init(args,config)
        
    #seed initialize
    print(f'Seed\t>>\t{args.seed}')
    torch_seed(args.seed)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = accelerator.device
    print(f'The device is ready\t>>\t{device}')

    #checkpoint_initialize
    checkpoint_path = util_tool.chechpoint_init(args)

    #transform for preprocessing input data
    transform, config = transform.get_transform(args)

    if args.use_wandb:
        wandb.config.update(config)                                

    dataset = ClassificationDataset(csv_path = args.csv_path,
                                    transform=transform)



    #(수정point)추후 factory로 함수화할 것
    n_train_set = int(args.train_val_split*len(dataset))
    train_set, val_set, train_idx, val_idx = sampler.train_valid_split_by_sklearn(dataset,args.train_val_split,args.seed)
    print(f'The number of training images\t>>\t{len(train_set)}')
    print(f'The number of validation images\t>>\t{len(val_set)}')
    #여기까지

    #(수정point)추후 factory로 자동화 할 것
    print('The data loader is ready ...')
    train_iter = DataLoader(train_set,
                            batch_size=args.batch_size,
                            drop_last=True,
                            num_workers=multiprocessing.cpu_count() // 2,
                            shuffle=True,
                            )   
    val_iter = DataLoader(val_set,
                          batch_size=args.batch_size,
                          drop_last=True,
                          num_workers=multiprocessing.cpu_count() // 2
                          )
    #


    print('The model is ready ...')
    model = Classifier(args).to(device)


    #(수정point)
    print('The optimizer is ready ...')
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    #(수정point)
    if args.lr_schduler:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    print('The loss function is ready ...')
    
    train_cnt = dataset.df['ans'][train_idx].value_counts().sort_index()
    normedWeights = [1 - (x / sum(train_cnt)) for x in train_cnt]
    normedWeights = torch.FloatTensor(normedWeights).to(device)
    
    #(수정point)준하님이 loss 완성하면 함수로 호출할 부분
    if args.loss == "crossentropy":
        criterion = nn.CrossEntropyLoss(normedWeights)
    elif args.loss == "focalloss":
        criterion = FocalLoss(alpha=0.1, device = device)
    
    #Accelerator 적용
    model, optimizer, train_iter, val_iter = accelerator.prepare(
        model, optimizer, train_iter, val_iter
    )

    best_val_f1 = 0 # saving the best F1 score along epoch
    logger = logging.getLogger('train')
    handler = logging.FileHandler(checkpoint_path + '/model.log')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for k, v in zip(args._fields, args):
        logger.info(k + ':' + str(v))

    train_tracker = util_tool.metric_tracker()
    val_tracker = util_tool.metric_tracker()

    print("Starting training ...")

    #(수정point) loss tracker로 loss나 metric을 따로 관리할 것
    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_iter_loss=0
        train_tracker.reset()
        pbar_train = tqdm(train_iter)
        for _,(train_img, train_target) in enumerate(pbar_train):
            pbar_train.set_description(f"Train. Epoch:{epoch}/{args.epochs} | Loss:{train_iter_loss:4.3f}")
            optimizer.zero_grad()
            train_pred = model(train_img)
            train_iter_loss = criterion(train_pred, train_target)
            accelerator.backward(train_iter_loss)
            optimizer.step()

            #train_tracker에 metric기록
            train_tracker.update_loss(train_iter_loss,len(train_target))
            train_tracker.update_cm_data([train_pred,train_target])

        pbar_train.close()

        #train_tracker 결과 가져오기
        train_epoch_loss = train_tracker.get_loss()

        # tracker.cm_data로 metric 계산
        train_cm = confusion_matrix2(train_tracker.get_cm_data(), args.num_classes)
        train_acc = accuracy(train_cm, args.num_classes)
        train_f1 = f1_score(train_cm, args.num_classes)

        # Validation
        with torch.no_grad():
            val_iter_loss = 0
            model.eval()
            pbar_val = tqdm(val_iter)
            val_tracker.reset()
            for _,(val_img, val_target) in enumerate(pbar_val):
                pbar_val.set_description(f"Val. Epoch:{epoch}/{args.epochs} | Loss:{val_iter_loss:4.3f}")
                val_pred = model(val_img)
                val_iter_loss = criterion(val_pred, val_target).detach()

                #val_tracker에 metric 기록
                val_tracker.update_loss(val_iter_loss,len(val_target))
                val_tracker.update_cm_data([val_pred, val_target])
                
            pbar_val.close()

        #val_tracker 결과 가져오기
        val_epoch_loss = val_tracker.get_loss()


        # val_cm = confusion_matrix(model, val_iter, device, args.num_classes)
        val_cm = confusion_matrix2(val_tracker.get_cm_data(), args.num_classes)

        val_acc = accuracy(val_cm, args.num_classes)
        val_f1 = f1_score(val_cm, args.num_classes)

        if args.lr_schduler:
            lr_scheduler.step()
        
        print('time >> {:.4f}\tepoch >> {:04d}\ttrain_acc >> {:.4f}\ttrain_loss >> {:.4f}\ttrain_f1 >> {:.4f}\tval_acc >> {:.4f}\tval_loss >> {:.4f}\tval_f1 >> {:.4f}'
              .format(time.time()-start_time, epoch, train_acc, train_epoch_loss, train_f1, val_acc, val_epoch_loss, val_f1))
        
        if best_val_f1 <= val_f1:
            if args.save_mode == 'state_dict' or args.save_mode == 'both':
                # 모델의 parameter들을 저장
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, os.path.join(checkpoint_path, f'best_state_dict.pt'))
            if args.save_mode == 'model' or args.save_mode == 'both':
                # 모델 자체를 저장
                torch.save(model, os.path.join(checkpoint_path, f'best_model.pt'))
            logger.info(f'epoch({epoch}), train_acc({train_acc:.3f}), train_loss({train_epoch_loss:.3f}), train_f1({train_f1:.3f}), val_acc({val_acc:.3f}), val_loss({val_epoch_loss:.3f}), val_f1({val_f1:.3f})')


        #추후 따로 뺴기
        if args.use_wandb:
            wandb.log({'Train Acc': train_acc,
                       'Train Loss': train_epoch_loss,
                       'Train F1-Score': train_f1,
                       'Val Acc': val_acc,
                       'Val Loss': val_epoch_loss,
                       'Val F1-Score': val_f1})

            if best_val_f1 <= val_f1:
                fig = plot.plot_confusion_matrix(val_cm, args.num_classes, normalize=True, save_path=None)
                wandb.log({'Confusion Matrix': wandb.Image(fig, caption=f"Epoch-{epoch}")})
                
        if best_val_f1 <= val_f1:
            best_val_f1 = val_f1



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="for model training")
    parser.add_argument('-c', '--config', type=str,default='./config.json', help='config file path for training. [default=\'./config.json\']')

    args = parser.parse_args()
    
    config = load_config(args.config)

    from collections import namedtuple
    Args = namedtuple('Args', config.keys())
    config = Args(**config)


    run(args, config)
    
    
