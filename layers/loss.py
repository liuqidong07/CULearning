# -*- encoding: utf-8 -*-
'''
@File    :   loss.py
@Time    :   2020/10/23 09:52:47
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch.nn as nn
from utils.util import *

#定义损失函数
class Loss(nn.Module):
    def __init__(self, loss_type='continuous', IPM_type=None):
        #loss_type:结果为连续变量的时候，使用MSE；为二值变量的时候，使用log loss。
        #loss_type={'continuous','binary'}
        #IPM_type={None, 'mmd', 'wass'}
        super().__init__()
        self.loss_type = loss_type
        self.ipm_type = IPM_type

    def forward(self, y, out_y):
        '''
        @y: 实际结果y
        @out_y: 模型预测结果
        @w: propensity score
        '''

        #计算预测误差prediction_loss
        if self.loss_type == 'continuous':
            loss = torch.mean(safe_sqrt((y - out_y) ** 2))
        elif self.loss_type == 'binary':
            out_y = 0.995 / (1.0 + torch.exp(-out_y)) + 0.0025
            loss = -torch.mean((y * safe_log(out_y) + (torch.ones_like(y) - y) * safe_log(torch.ones_like(out_y) - out_y)))
        else:
            raise Exception("No valid loss type!")

        return loss