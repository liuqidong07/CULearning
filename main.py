# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/10/21 10:24:12
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib

import argparse
from models.CULearning import CausalUnlabeled
from preprocess.preprocess import create_train_test
from data.dataset import *
from utils import evaluation
from layers.Loss import Loss
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

'''input parameters from shell'''
parser = argparse.ArgumentParser(description='train and test models')
parser.add_argument('-m', default='tarnet', choices=['bnn', 'tarnet', 'cfr', 'dragonnet', 'scrnet', 'dr_cfr', 'der_cfr'], help='choose model')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-lr-decay', default=0.97, type=float, help='learning rate decay')
parser.add_argument('-lr-type', default='exp', choices=['exp', 'step', 'cos'], help='method of learning rate decay')
parser.add_argument('-period', default=100, type=int, help='how many epochs to conduct a learning rate decay')
parser.add_argument('-epoch', default=80, type=int, help='the number of epoch')
parser.add_argument('-bs', default=100, type=int, help='batch size')
parser.add_argument('-es', default=1, type=bool, help='early stop or not')
parser.add_argument('-dataset', default='ihdp', choices=['ihdp', 'jobs', 'news', 'twins'], help='choose dataset')
parser.add_argument('-optimizer', default='adam', choices=['sgd', 'adam', 'rmsprop'], help='choose the optimization of training')
parser.add_argument('-bn', default=0, type=bool, help='batch normalization or not')
parser.add_argument('-init', default='normal', choices=['normal', 'kaiming'], help='how initialize the neural network')
parser.add_argument('-train-test', default=0.9, type=float, help='the ratio of training size in whole dataset')
parser.add_argument('-train-val', default=0.7, type=float, help='the ratio of training size in training dataset')
parser.add_argument('-scale', default=1, type=int, help='wheather scale the x of dataset')

parser.add_argument('-alpha', default=1, type=float, help='the weight of imbalance loss in DR-CFR. the weight of adjustment decomposing loss in DeR-CFR.')
parser.add_argument('-beta', default=1, type=float, help='the weight of cross entropy loss to precisely predict treatment in DR-CFR. the weight of instrument decomposing loss in DeR-CFR.')
parser.add_argument('-gamma', default=1, type=float, help='the weight of confounder balance loss in DeR-CFR')
parser.add_argument('-mu', default=1, type=float, help='the weight of orthogonal regularizer for hard decomposition in DeR-CFR.')
parser.add_argument('-lamda', default=0.001, type=float, help='the weight of parameter regularization')

parser.add_argument('-ipm', default='wass', choices=['mmd', 'wass', 'nothing'], help='the empirical integral probability matrix')
parser.add_argument('-r-dim', default=200, type=int, help='representation layer dimension')
parser.add_argument('-h-dim', default=100, type=int, help='the inference layer dimension')
parser.add_argument('-rep', default=10, type=int, help='how many replications do you want to conduct')
parser.add_argument('-log', default=1, type=bool, help='whether log the experiment.')
parser.add_argument('-log-path', default='default', choices=['default', 'time'], help='where is the log')
parser.add_argument('-draw', default=0, type=int, help='wheather draw the process of training')


def run(args):
    '''load train and test data'''
    inDict = get_inputDict()
    trainData = schoolTrain()
    testData = schoolTest()
    n = 54
    

    '''instantiate model'''
    model = CausalUnlabeled(n, inDict)
    model = train_model(model, trainData, args)

    testLoader = DataLoader(testData, batch_size=testData.__len__())
    for batch in testLoader:
        t, x_cont, x_cate, y = batch[0], batch[1], batch[2], batch[3]
        y_pred = model(x_cont, x_cate, t)
        y_pred = y_pred.round()    #做四舍五入
        acc_test = evaluation(y, y_pred)
    print('testset acc: %f' % acc_test)
    


def train_model(model, trainData, args):

    '''slit trainset and validation set'''
    train_size = int(args.train_test * trainData.__len__())
    val_size = trainData.__len__() - train_size
    trainData, valData = random_split(dataset=trainData, lengths=[train_size, val_size])
    val_loader = DataLoader(valData, batch_size=val_size)
    batch_num = int(trainData.__len__() / args.bs) + 1

    '''create optimizer'''
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for val_batch in val_loader:
        pass

    train_loss, train_acc = 0, 0
    '''start train'''
    for epoch in range(args.epoch):
        train_loader = DataLoader(trainData, batch_size=args.bs, shuffle=True)
        for batch in train_loader:
            t, x_cont, x_cate, y = batch[0], batch[1], batch[2], batch[3]
            y_pred = model(x_cont, x_cate, t)
            criterion = Loss(loss_type='binary')
            loss = criterion(y, y_pred)
            train_loss += loss / batch_num
            train_acc += evaluation.accuracy(y.detach().numpy(), y_pred.round().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
        '''compute loss and acc on validation set'''
        y_pred_val = model(val_batch[1], val_batch[2], val_batch[0])
        criterion = Loss(loss_type='binary')
        val_loss = criterion(val_batch[3], y_pred_val)
        val_acc = evaluation.accuracy(val_batch[3].detach().numpy(), y_pred_val.round().detach().numpy())
        print('--epoch: %d/%d:--' % (epoch, args.epoch))
        print('\t train loss: %f, train acc: %f' % (train_loss, train_acc))
        print('\t val loss: %f, val acc: %f\n' % (val_loss, val_acc))
    print('training complete')

    return model


if __name__ == '__main__':
    args = parser.parse_args()
    run(args=args)







