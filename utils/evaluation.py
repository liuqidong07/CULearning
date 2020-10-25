# -*- encoding: utf-8 -*-
'''
@File    :   evaluation.py
@Time    :   2020/10/22 14:59:14
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch
import numpy as np
from sklearn.metrics import accuracy_score

def accuracy(tlabel, plable):
    '''
    compute the accuracy of prediction
    @ tlabel: true label
    @ plabel: predicted label
    '''
    return accuracy_score(tlabel, plable)




