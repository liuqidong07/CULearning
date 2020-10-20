# -*- encoding: utf-8 -*-
'''
@File    :   activation.py
@Time    :   2020/10/19 15:39:55
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch.nn.functional as F

'''Define your own activation layers here!'''
def no_act(x):
    return x


'''select the activation function'''
def activation(selection):
    """
    select the activation function
    @ selection: str, select the activation function
    """
    if selection.lower() == 'none':
        return no_act
    elif selection.lower() == 'sigmoid':
        return F.Sigmoid
    elif selection.lower() == 'relu':
        return F.ReLU
    elif selection.lower() == 'elu':
        return F.ELU
    elif selection.lower() == 'softmax':
        return F.Softmax
    else:
        raise Exception('Please implement this activation!')

    







