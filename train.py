import torch
from torch import nn
from torch.utils.data import dataloader

import dataloader as L
from metric import loss
from metric.optimizer import set_optimizer
from util import logger
import importlib
from tqdm import tqdm
from torchsummary import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer():
    def __init__(self,model, config):
        self.config = config
        self.model = model
        loader_mod = importlib.import_module('dataloader.'+
                                             config['dataset']['name'])
        dataset_name = L.convert_dataset_name(config['dataset']['name'])
        dataset_init = getattr(loader_mod,dataset_name)
        self.dataset = dataset_init(**config['dataset']['args'])
        self.dataloader = dataloader.DataLoader(self.dataset,
                                                      **config['dataloader'])
        
        self.criterion = self.model.criterion
        self.EPOCH = config['epochs']
        self.batch_size = config['dataloader']['batch_size']
        self.optim_name = config['optimizer']['name']
        self.lr = config['optimizer']['args']['lr']
        self.backbone_name = config['backbone']['name']
        self.optim = set_optimizer(self.model.parameters(),
                                    config['optimizer']['name'],
                                    config['optimizer']['args'])
        
        self.train_loss = loss.loss_tracker()
        self.val_loss = loss.loss_tracker()
        self.logger = logger.logger(self.batch_size,self.EPOCH,
                                    self.optim_name,
                                    self.lr,
                                    self.backbone_name)
        self.valid = False
        self.save_name = config['save_name']
        #summary(self.model.to(device),(3,224,224))
        print('Using device: ',device)
        
    
        
        #test_getitem(self.dataloader)

    def run(self):
        self.model.to(device)
        torch.save(self.model.state_dict(),f'./saved_model/{self.save_name}')
        for epoch in range(self.EPOCH):
            self.model.train()
            self.dataset.set_train_mode()
            self.train_loss.reset()
            self.val_loss.reset()
            train_acc = 0
            train_cnt = 0
            with tqdm(self.dataloader) as pbar:
                print(str(epoch)+'train')
                for idx,data in enumerate(pbar):
                    x,label = data
                    self.optim.zero_grad()
                    logit = self.model(x.to(device))
                    loss = self.criterion(logit,label.to(device))
                    loss.backward()
                    self.train_loss.update(loss,self.batch_size)
                    self.optim.step()
                    train_acc += self.model.accuracy(logit,label.to(device))
                    train_cnt += self.batch_size
                    pbar.set_description(f'Epoch:{epoch}/{self.EPOCH},'+\
                                         f' cur_loss/avg_loss:{loss.item()/self.batch_size:4.3f}/{self.train_loss.get_loss():4.3f}'+\
                                            f', acc: {train_acc}/{train_cnt}')
                    
            
            
            print(logit.argmax(dim=1))
            print(label)
            self.model.eval()
            self.dataset.set_val_mode()
            val_acc = 0
            val_cnt = 1
            if self.valid:
                with tqdm(self.dataloader) as pbar:
                    print(str(epoch)+'validation')
                    for idx,data in enumerate(pbar):
                        x,label = data
                        logit = self.model(x.to(device))
                        loss = self.criterion(logit,label.to(device))
                        self.val_loss.update(loss,self.batch_size)
                        val_acc += self.model.accuracy(logit,label.to(device))
                        val_cnt += self.batch_size
                        pbar.set_description(f'Epoch:{epoch}/{self.EPOCH}, cur_loss/avg_loss:'\
                            +f'{loss.item()/self.batch_size:4.3f}/{self.val_loss.get_loss():4.3f}'+\
                                f', acc: {val_acc}/{val_cnt}')
                    
            #print(logit,label)
            self.logger.write({'Train Acc':train_acc/train_cnt, "Train Loss":self.train_loss.get_loss(),
                         'Val Acc':val_acc/val_cnt, 'Val loss':self.val_loss.get_loss()})

            #print(logit,label)

        torch.save(self.model,f'./saved_model/{self.save_name}')

def test_getitem(dataloader):
    idx,data = next(enumerate(dataloader))
    x,y = data
    print(x.shape)
    print(y)
    print(y.shape)