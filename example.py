import logging
import time

from collections import namedtuple

args_dict = {'seed' : 223,
            'csv_path' : '../input/data/train/train_info.csv',
            'save_path' : './checkpoint',
            'num_classes' : 18,
            'num_mask_classes' : 3,
            'num_gender_classes' : 2,
            'num_age_classes' : 3,
            'model_summary' : True,
            'batch_size' : 64,
            'learning_rate' : 5e-6,
            'epochs' : 100,
            'train_val_split': 0.8,
            'save_mode' : 'state_dict', #'model'
            'save_epoch' : 10,
            'load_model': 'vit_small_patch16_224', #'densenet121', #'resnet18',
            'loss' : "crossentropy",
            'transform_path' : './utils/transform_list.json',
            'transform_list' : ['resize', 'randomhorizontalflip', 'randomrotation', 'totensor', 'normalize'],#['resize', 'randomhorizontalflip', 'randomrotation', 'totensor', 'normalize'],
            'not_freeze_layer' : ['layer4'],
            'weight_decay': 5e-4}
Args = namedtuple('Args', args_dict.keys())
args = Args(**args_dict)

path = '.'
logger = logging.getLogger('example')
fh = logging.FileHandler(path + '/example.log')
logger.setLevel(logging.INFO)
logger.addHandler(fh)

logger.info(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
# logger.info('%s', args)
for k, v in zip(args._fields, args):
    # print(k, v)
    logger.info(k + ':' + str(v))