'''
@Author: NoserQJH
@LastEditors  : NoserQJH
@Date: 2019-12-27 16:39:44
@LastEditTime : 2019-12-27 17:21:34
@Description:
'''

import torchvision as tv
import torch
from config import get_config


def get_train_loader(config, dataset='SVHN'):
    if dataset == 'SVHN':
        data_train = tv.datasets.SVHN(
            './Datasets/'+dataset, split='train', download=True)
    elif dataset == 'USPS':
        data_train = tv.datasets.USPS(
            './Datasets/'+dataset, train=True, download=True)
    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=config.batch_size,
                                                    shuffle=True,
                                                    num_workers=config.num_workers)
    return data_train_loader


def get_test_loader(config, dataset='SVHN'):
    if dataset == 'SVHN':
        data_test = tv.datasets.SVHN(
            './Datasets/'+dataset, split='test', download=True)
    elif dataset == 'USPS':
        data_test = tv.datasets.USPS(
            './Datasets/'+dataset, train=False, download=True)
    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                   batch_size=config.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.num_workers)
    return data_test_loader

def main():
    conf = get_config()[0]
    train_loader = get_train_loader(conf, 'USPS')
    test_loader = get_test_loader(conf, 'USPS')
    print('Done')

if __name__ == '__main__':
    main()

