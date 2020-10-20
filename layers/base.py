# -*- encoding: utf-8 -*-
'''
@File    :   base.py
@Time    :   2020/10/19 15:27:32
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch
import torch.nn as nn
from layers.activation import activation


class DNN(nn.Module):
    '''
    create deep neural networks with specified hidden units and layer number
    '''
    def __init__(self, input_dim, output_dim, hidden_layers=3, 
        hidden_units=64, drop_rate=0.5, activation='relu') -> None:
        super(DNN, self).__init__()
        '''
        @ input_dim: int, the dimension of input layers
        @ output_dim: int, the dimension of output layers
        @ hidden_layers: int, the number of hidden layers
        @ hidden_units: int,  the number of units in hidden layers
        @ drop_out: bool, dropout or not
        '''
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.inLinear = nn.Linear(input_dim, hidden_units)
        self.linearGroup = nn.ModuleList([nn.Linear(hidden_units, hidden_units) for _ in range(hidden_layers)])
        self.outLinear = nn.Linear(hidden_units, output_dim)
        self.dropGroup = nn.ModuleList([nn.Dropout(p = drop_rate) for _ in range(hidden_layers)])

    def forward(self, x):
        x = activation(selection=activation)(self.inLinear(x))
        for i in range(self.hidden_layers):
            x = activation(selection=activation)(self.dropGroup[i](self.linearGroup[i](x)))
        y = activation(selection=activation)(self.outLinear(x))
        return y





