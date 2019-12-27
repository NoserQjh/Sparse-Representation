# -*- coding: utf-8 -*-
'''
@Author: NoserQJH
@LastEditors  : NoserQJH
@Date: 2019-12-11 21:37:14
@LastEditTime : 2019-12-27 17:16:15
@Description:
'''

import argparse

arg_lists = []
parser = argparse.ArgumentParser(description='mobilenet_classification')

parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=4)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

conf = get_config()[0]
print(conf)
