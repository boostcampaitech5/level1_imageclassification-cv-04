from dataloader import *
from model import *
from metric import *
from utils import *
from torch.utils.data import DataLoader, random_split
from torchvision import transforms 
from torch import optim
import numpy as np
import random
import os
import wandb
from torchsummary import summary
import time
import multiprocessing

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
    if args.use_wandb:
        print('Initialize WandB ...')
        wandb.init(name = args.wandb_exp_name,
                   project = args.wandb_project_name,
                   entity = args.wandb_entity,
                   config = args_dict)
        
    torch_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'The device is ready\t>>\t{device}')

    print('Make save_path')
    os.makedirs(args.save_path, exist_ok=True)


    transform = transforms.Compose([transforms.Resize((args.img_width, args.img_height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])
                                    # transforms.Normalize((0.5601, 0.5241, 0.5014),
                                    #                      (0.2331, 0.2430, 0.2456))])

    # transform = albumentations.Compose([transforms.Resize((args.img_width, args.img_height)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize((0.485, 0.456, 0.406),
    #                                                          (0.229, 0.224, 0.225))])

    dataset = ClassificationDataset(csv_path = args.csv_path,
                                    transform=transform)
    
    n_train_set = int(args.train_val_split*len(dataset))
    train_set, val_set = random_split(dataset, [n_train_set, len(dataset)-n_train_set])
    print(f'The number of training images\t>>\t{len(train_set)}')
    print(f'The number of validation images\t>>\t{len(val_set)}')

    train_iter = DataLoader(train_set,
                            batch_size=args.batch_size,
                            shuffle = True,
                            drop_last=True,
                            num_workers=multiprocessing.cpu_count() // 2,)
    val_iter = DataLoader(val_set,
                          batch_size=args.batch_size,
                          shuffle = True,
                          drop_last=True,
                          num_workers=multiprocessing.cpu_count() // 2,)
    

    print('The model is ready ...')
    model = Classifier(args.num_classes, args.load_model).to(device)
    if args.model_summary:
        print(summary(model, (3, 256, 256)))

    print('The optimizer is ready ...')
    optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate)

    print('The loss function is ready ...')
    criterion = nn.CrossEntropyLoss()

    print("Starting training ...")
    for epoch in range(args.epochs):
        start_time = time.time()
        train_epoch_loss = 0
        model.train()
        for train_img, train_target in train_iter:
            train_img, train_target = train_img.to(device), train_target.to(device)
            
            optimizer.zero_grad()

            train_pred = model(train_img)
            train_iter_loss = criterion(train_pred, train_target)
            train_iter_loss.backward()
            optimizer.step()

            train_epoch_loss += train_iter_loss

        train_epoch_loss = train_epoch_loss / len(train_iter)

        train_cm = confusion_matrix(model, train_iter, device, args.num_classes)

        train_acc = accuracy(train_cm, args.num_classes)
        train_f1 = f1_score(train_cm, args.num_classes)

        # Validation
        with torch.no_grad():
            val_epoch_loss = 0
            model.eval()
            for val_img, val_target in val_iter:
                val_img, val_target = val_img.to(device), val_target.to(device)

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
                    }, os.path.join(args.save_path, f'epoch({epoch})_acc({val_acc:.3f})_loss({val_epoch_loss:.3f})_f1({val_f1:.3f})_state_dict.pt'))
            if args.save_mode == 'model' or args.save_mode == 'both':
                # 모델 자체를 저장
                torch.save(model, os.path.join(args.save_path, f'epoch({epoch})_acc({val_acc:.3f})_loss({val_epoch_loss:.3f})_f1({val_f1:.3f})_model.pt'))

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

if __name__ == '__main__':
    args_dict = {'seed' : 223,
                 'csv_path' : './input/data/train/train_info.csv',
                 'save_path' : './checkpoint',
                 'use_wandb' : True,
                 'wandb_exp_name' : 'exp1_bs64_ep100_adam_lr0.0001_resnet50.jy',
                 'wandb_project_name' : 'Image_classification_mask',
                 'wandb_entity' : 'connect-cv-04',
                 'num_classes' : 18,
                 'model_summary' : True,
                 'batch_size' : 64,
                 'learning_rate' : 1e-4,
                 'epochs' : 100,
                 'train_val_split': 0.8,
                 'save_mode' : 'both',
                 'save_epoch' : 10,
                 'load_model':'resnet50',
                 'img_width' : 256,
                 'img_height' : 256}
    
    from collections import namedtuple
    Args = namedtuple('Args', args_dict.keys())
    args = Args(**args_dict)

    # Config parser 하나만 넣어주면 됨(임시방편)
    run(args, args_dict)