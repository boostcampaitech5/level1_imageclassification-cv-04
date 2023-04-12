import torch
from torch import nn
from torch.utils.data import dataloader

import dataloader as L
from metric import loss
from metric.optimizer import set_optimizer
import importlib
from tqdm import tqdm

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

        self.optim = set_optimizer(self.model.parameters(),
                                    config['optimizer']['name'],
                                    config['optimizer']['args'])
        
        self.train_loss = loss.loss_tracker()
        self.val_loss = loss.loss_tracker()
        
        print('Using device: ',device)
        
    
        
        #test_getitem(self.dataloader)

    def run(self):
        self.model.to(device)
        for epoch in range(self.EPOCH):
            self.model.train()
            self.dataset.set_train_mode()
            self.train_loss.reset()
            self.val_loss.reset()
            with tqdm(self.dataloader) as pbar:
                for idx,data in enumerate(pbar):
                    x,label = data
                    self.optim.zero_grad()
                    logit = self.model(x.to(device))
                    loss = self.criterion(logit,label.to(device))
                    loss.backward()
                    self.train_loss.update(loss,self.batch_size)
                    self.optim.step()
                    pbar.set_description(f'Epoch:{epoch}/{self.EPOCH}, cur_loss/avg_loss:\
                        {loss}/{self.train_loss.get_loss()}')
            self.model.eval()
            self.dataset.set_val_mode()
            with tqdm(self.dataloader) as pbar:
                for idx,data in enumerate(pbar):
                    x,label = data
                    logit = self.model(x.to(device))
                    loss = self.criterion(logit,label.to(device))
                    self.val_loss.update(loss,self.batch_size)
                    
                    pbar.set_description(f'Epoch:{epoch}/{self.EPOCH}, cur_loss/avg_loss:\
                        {loss}/{self.val_loss.get_loss()}')
                print(logit,label)

def test_getitem(dataloader):
    idx,data = next(enumerate(dataloader))
    x,y = data
    print(x.shape)
    print(y)
    print(y.shape)