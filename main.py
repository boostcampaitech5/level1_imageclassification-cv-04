import test
from train import Trainer
from util import *
import model as M
import dataloader as L

import argparse

def run(config):
    model_config = config['model']
    model = M.get_model(**model_config)
    dataset = L.get_dataset(config['dataset'])
    train_loader, valid_loader = L.get_loader(config['dataloader'],dataset)

    return 
    trainer = Trainer(model,dataset)
    tester = None
    trainer.train()



if __name__ == '__main__':
    config = read_json('sample_config.json')
    run(config)