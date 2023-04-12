import test
from train import Trainer
from util import *
import model as M

from metric import loss

import argparse

def run(config):
    model_config = config['model']
    model = M.get_model(**model_config)

    trainer = Trainer(model,config)
    trainer.run()



if __name__ == '__main__':
    config = read_json('config.json')
    run(config)