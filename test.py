import logging
import time
import os
import json
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from utils.util import plot_confusion_matrix, toConfusionMatrix
from train import AverageMeter, cmMetter, outputToPred
from torch.utils.data import DataLoader
from torchvision import transforms
import multiprocessing
from tqdm import tqdm

_logger = logging.getLogger('test')

def test(model, testloader, accelerator, savedir, args) -> dict:   

    source_csv_path = os.path.join(args.datadir, args.test_file)
    target_csv_path = os.path.join(args.datadir, f'test_{args.exp_name}_{args.exp_num}.csv')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'The device_age is ready\t>>\t{device}')
  
    print('Loading the model ...')
    model = model

    print('Loading checkpoint ...')
    state_dict = torch.load(os.path.join(savedir, f'best_model.pt'))
    model.load_state_dict(state_dict)

    print("Starting testing ...")
    model.eval()
    result = []

    pbar_test = tqdm(testloader)
    for _, (test_img, _) in enumerate(pbar_test):
        pbar_test.set_description(f"Test. iter:")
        with torch.no_grad():
            test_img = test_img.to(device)
            test_pred = torch.max(model(test_img), 1)[1]
            result.append(test_pred.item())
    pbar_test.close()

    df = pd.read_csv(source_csv_path)
    df['ans'] = result
    df.to_csv(target_csv_path, index=False)
    print('Save CSV file', target_csv_path)
