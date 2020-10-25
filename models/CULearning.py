# -*- encoding: utf-8 -*-
'''
@File    :   CULearning.py
@Time    :   2020/10/19 10:40:02
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from layers.activation import activation
import torch
import torch.nn as nn
from layers import inputs, base


class CausalUnlabeled(torch.nn.Module):
    def __init__(self, n, input_dict, em_dim=8, r_dim=32, rh_layers=2, 
        rh_dim=32, rh_drop=0.5, ph_layers=1, ph_dim=16, ph_drop=0.5, 
        activation='relu') -> None:
        '''
        @ n: the number of head for predicting outcomes respectively
        @ input_dict: the dict contains the information of raw input, e.g. {'cont': int, 'cate': [cate1, cate2, ...]} 
        @ em_dim: the output dimension of embedding layer
        @ r_dim: the dimension of representation
        @ rh_layers: the number of hidden layers in representation
        @ rh_dim: the dimension of hidden layers in representation
        @ rh_drop: dropout rate in representation
        @ rh_layers: the number of hidden layers in prediction
        @ rh_dim: the dimension of hidden layers in prediction
        @ rh_drop: dropout rate in prediction
        @ activation: select activation function
        '''
        super(CausalUnlabeled, self).__init__()
        self.n_head = n
        self.in_ = inputs.LowFeature(input_dict, em_dim)
        low_dim = em_dim * len(input_dict['cate']) + input_dict['cont']    #compute low feature dim
        self.rep = base.DNN(low_dim, r_dim, rh_layers, rh_dim, rh_drop, activation)
        self.head = nn.ModuleList([base.DNN(r_dim, 1, ph_layers, ph_dim, ph_drop, activation) for _ in range(n)])



    def forward(self, x_cont, x_cate, t):
        x = self.in_(x_cont, x_cate)
        x = self.rep(x)
        y = [self.head[t[i][0]](x[i]) for i in range(len(t))]
        return torch.autograd.Variable(torch.Tensor(y), requires_grad=True)








