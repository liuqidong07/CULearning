# -*- encoding: utf-8 -*-
'''
@File    :   PULearning.py
@Time    :   2020/10/15 16:20:10
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import numpy as np
import pandas as pd
from preprocess import preprocess

train_path = r'./data/train.csv'
test_path = r'./data/test.csv'

class bagging():
    def __init__(self, P, U, K, T) -> None:
        self.pdata = P
        self.udata = U
        self.K = K
        self.T = T




