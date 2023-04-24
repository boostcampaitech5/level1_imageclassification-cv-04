import os
from torch.utils.data import DataLoader


def create_dataset(datadir: str, dataname: str, aug_name: str = 'default'):
    trainset = __import__('torchvision.datasets', fromlist='datasets').__dict__[dataname](
        root      = os.path.join(datadir,dataname), 
        train     = True, 
        download  = True, 
        transform = __import__('datasets').__dict__[f'{aug_name}_augmentation']()
    )
    testset = __import__('torchvision.datasets', fromlist='datasets').__dict__[dataname](
        root      = os.path.join(datadir,dataname), 
        train     = False, 
        download  = True, 
        transform = __import__('datasets').__dict__['test_augmentation']()
    )

    return trainset, testset


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 2
    )
