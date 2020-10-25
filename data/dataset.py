# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2020/10/22 15:27:15
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

train_path = r'./data/train.csv'
test_path = r'./data/test.csv'

cont = ['researchExp', 'industryExp', 'toeflScore', 
    'program', 'internExp', 'greV', 'greQ', 'journalPubs', 
    'greA', 'confPubs', 'topperCgpaScale', 'CgpaScale']
cate = ['userName', 'major', 'department', 'ugCollege', 
    'termAndYear', 'specialization']

class schoolTrain(Dataset):
    '''
    
    '''
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv(train_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        x_cont = self.data[cont].iloc[index, :]
        x_cate = self.data[cate].iloc[index, :]
        y = self.data['admit'].iloc[index]
        t = self.data['univName'].iloc[index]
        return np.array([t]), np.array(x_cont.tolist()), np.array(x_cate.tolist()), np.array([y])


class schoolTest(Dataset):
    '''
    
    '''
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv(test_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        x_cont = self.data[cont].iloc[index, :]
        x_cate = self.data[cate].iloc[index, :]
        y = self.data['admit'].iloc[index]
        t = self.data['univName'].iloc[index]
        return t, x_cont.tolist(), x_cate.tolist(), y


def get_inputDict():
    '''
    get input dict {'cont': int(cont feature dim), 'cate': [f1_dim, f2_dim, ...]}
    '''
    data = pd.read_csv(r'./data/handled_data.csv')
    inDict = {}
    inDict['cont'] = len(cont)
    inDict['cate'] = data[cate].nunique()
    return inDict
