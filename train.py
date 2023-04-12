from dataloader import *
from model import *
from metric import *
from torch.utils.data import DataLoader
from torchvision import transforms 
from torch import optim
import numpy as np
import random
import os
import wandb
from torchsummary import summary
import time

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

def run(cfg):
    if cfg.use_wandb:
        print('Initialize WandB ...')
        wandb.init(name = cfg.wandb_exp_name,
                   project = cfg.wandb_project_name,
                   entity = cfg.wandb_entity,
                   config = cfg)
        
    torch_seed(cfg.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'The device is ready\t>>\t{device}')

    transform = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.ToTensor()])

    dataset = ClassificationDataset(csv_path = cfg.csv_path,
                                    transform=transform,
                                    num_classes = cfg.num_classes)
    print(f'The number of training images\t>>\t{len(dataset)}')

    train_iter = DataLoader(dataset,
                            batch_size=cfg.batch_size,
                            shuffle = True)
    

    print('The model is ready ...')
    model = Network(num_classes = cfg.num_classes).to(device)
    if cfg.model_summary:
        print(summary(model, (3, 256, 256)))

    print('The optimizer is ready ...')
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.learning_rate)

    print('The loss function is ready ...')
    criterion = nn.CrossEntropyLoss()

    print("Starting training ...")
    for epoch in range(cfg.epochs):
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

        cm = confusion_matrix(model, train_iter, device, cfg.num_classes)
        train_acc = accuracy(cm, cfg.num_classes)
        train_f1 = f1_score(cm, cfg.num_classes)

        print('time >> {:.4f}\tepoch >> {:04d}\ttrain_acc >> {:.4f}\ttrain_loss >> {:.4f}\ttrain_f1 >> {:.4f}'
              .format(time.time()-start_time, epoch, train_acc, train_epoch_loss, train_f1))
        
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, cfg.save_path / f'epoch({epoch})_acc({train_acc:.3f})_loss({train_epoch_loss:.3f})_f1({train_f1:.3f}).pt')
        
        if cfg.use_wandb:
            wandb.log({'train_acc': train_acc,
                       'train_loss': train_epoch_loss,
                       'train_f1': train_f1})


if __name__ == '__main__':
    class cfg:
        seed = 223
        csv_path = './input/data/train/train_info.csv'
        save_path = './checkpoint'
        use_wandb = False
        wandb_exp_name = 'test'
        wandb_project_name = 'Image_classification_mask'
        wandb_entity = 'connect-cv-04'
        num_classes = 18
        model_summary = False
        batch_size = 3
        learning_rate = 1e-4
        epochs = 10

    run(cfg)