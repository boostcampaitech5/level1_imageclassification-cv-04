import test
from train import Trainer
from util import *
import model as M
import dataloader as L

import argparse

def run(config):
    model_config = config['model']
    model = M.get_model(**model_config)
    train_dataset = L.get_dataset(config['dataset'])
    train_dataloader = L.get_loader(config['dataloader'],train_dataset)

    trainer = Trainer(model,train_dataloader)
    tester = None
    trainer.train()



if __name__ == '__main__':
    config = read_json('sample_config2.json')
    run(config)