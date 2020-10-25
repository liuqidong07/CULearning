# -*- encoding: utf-8 -*-
'''
@File    :   util.py
@Time    :   2020/10/21 10:27:55
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch


def safe_sqrt(tensor):
    return torch.sqrt(tensor + 1e-8)


def safe_log(tensor):
    return torch.log(tensor + 1e-8)


