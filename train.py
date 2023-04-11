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
        make_dataset = getattr(loader_mod,'make_dataset')
        self.train_set, self.val_set = make_dataset(config['dataset'])
        self.train_dataloader = dataloader.DataLoader(self.train_set,
                                                      **config['dataloader'])
        self.val_dataloader = dataloader.DataLoader(self.val_set,
                                                      **config['dataloader'])
        self.criterion = loss.set_loss(config['loss'])
        self.EPOCH = config['epochs']

        self.optim = set_optimizer(self.model.parameters(),
                                    config['optimizer']['name'],
                                    config['optimizer']['args'])

        print('Using device: ',device)
        

    def run(self):
        self.model.train()
        self.model.to(device)
        for epoch in range(self.EPOCH):
            with tqdm(self.train_dataloader) as pbar:
                for idx,data in enumerate(pbar):
                    pbar.set_description(f'Epoch:{epoch}/{self.EPOCH}')
                    x,label = data
                    self.optim.zero_grad()
                    logit = self.model(x)
                    loss = self.criterion(logit,label)
                    loss.backward()
                    self.optim.step()
                    

import time