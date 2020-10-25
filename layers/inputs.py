# -*- encoding: utf-8 -*-
'''
@File    :   inputs.py
@Time    :   2020/10/19 11:40:48
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch
from torch import long
import torch.nn as nn

class LowFeature(nn.Module):
    '''
    extract low features from raw data
    '''
    def __init__(self, input_dict, embedding_dim=8) -> None:
        '''
        @ input_dict: contain continuous variable dimension and categorical variable dimension
        @ input_dict['cont']: int
        @ input_dict['cate']: list
        @ embedding_dim: the output dimension of each embedding layer
        '''
        super(LowFeature, self).__init__()
        self.cont_dim = input_dict['cont']
        self.cate_num = len(input_dict['cate'])
        self.em_dim = input_dict['cate']
        #TODO: 一般情况下embedding的输出维度是类别数的0.25倍？
        self.em = nn.ModuleList([nn.Embedding(cdim, embedding_dim) for cdim in input_dict['cate']])



    def forward(self, x_cont, x_cate):
        """
        @ x_cont: the input of continuous variables
        @ x_cate: the input of categorical variables
        """
        em_cate = self.em[0](x_cate[:, 0].view(-1, 1).long()).squeeze(1)
        '''concate all embedding vactor of categorical variable'''
        for i in range(1, self.cate_num):
            em_cate = torch.cat((em_cate, self.em[i](x_cate[:, i].view(-1, 1).long()).squeeze(1)), 1)
        y = torch.cat((x_cont.float(), em_cate), 1)

        return y





