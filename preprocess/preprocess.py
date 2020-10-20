# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2020/10/10 14:58:32
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

'''
output train_data.csv and test_data.csv
train_data.csv only contains positive data samples for PU Learning
test_data.csv contains positive and negative samples for evaluation 
'''


# here put the import lib
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

original_path = r'./data/original_data.csv'
transform_path = r'./data/score.csv'
out_path = r'./data/handled_data.csv'
train_path = r'./data/train.csv'
test_path = r'./data/test.csv'

def preprocess():
    '''
    preprocess the original data, including 
    (1) dropping some useless features,
    (2) transforming GRE scores from old to new,
    (3) filling null data,
    (4) scaling every useful features,
    (5) and converting categorical feature to one-hot.
    return the handled dataframe
    '''
    data = pd.read_csv(original_path)
    score_table = pd.read_csv(transform_path)
    
    '''drop useless features'''
    data = data.drop(['userProfileLink', 'gmatA', 'gmatQ', 'gmatV', 'toeflEssay'], 1)    #1代表删除的是列

    '''fill numerate null data with 0, fill character null data with string NULL'''
    data[['toeflScore', 'internExp', 'greV', 'greQ', 'greA', 'journalPubs', 'confPubs']] = data[['toeflScore', 'internExp', 'greV', 'greQ', 'greA', 'journalPubs', 'confPubs']].fillna(value=0)
    data[['major', 'specialization', 'program', 'termAndYear', 'ugCollege']] = data[['major', 'specialization', 'program', 'termAndYear', 'ugCollege']].fillna(value='NULL')
    data = data.dropna()    #drop the column contained with 0

    '''transform GRE scores from old to new'''
    data['greA'] = scoreConversion('greA', data, score_table)
    data['greQ'] = scoreConversion('greQ', data, score_table)

    '''scale features'''
    '''scale gpa'''
    #data['gpaScale'] = data['cgpa'] / data['topperCgpa']
    #data = data.drop(columns=['cgpa', 'topperCgpa'])
    '''Z-score'''
    temp = data[['greQ', 'greV', 'toeflScore']]
    data[['greQ', 'greV', 'toeflScore']] = (temp - temp.mean()) / temp.std()
    '''convert to 0-1'''
    data['researchExp'] = np.where(data['researchExp'] == 0, 0, 1)
    data['industryExp'] = np.where(data['industryExp'] == 0, 0, 1)
    data['specialization'] = np.where(data['specialization'] == 0, 0, 1)
    data['internExp'] = np.where(data['internExp'] == 0, 0, 1)
    data['greA'] = np.where(data['greA'] == 0, 0, 1)
    data['journalPubs'] = np.where(data['journalPubs'] == 0, 0, 1)
    data['confPubs'] = np.where(data['confPubs'] == 0, 0, 1)

    '''handle categorical feature'''
    '''label encoding'''
    encoder = LabelEncoder()
    encoded = data[['userName', 'major', 'program', 'department', 'termAndYear', 'ugCollege', 'univName']].apply(encoder.fit_transform)
    data[['userName', 'major', 'program', 'department', 'termAndYear', 'ugCollege', 'univName']] = encoded
    
    #TODO: categorical feature encoding

    return data


def create_train_test():
    '''
    we want to create a dataset only contain positive labelled sample.
    so, we randomly select one admitted apply of each person which make up the training set
    all other samples goes into test set(because they are ground truth)
    '''

    '''if data is preprocessed, read it, otherwise create it'''
    if os.path.isfile(out_path):
        data = pd.read_csv(out_path)
    else:
        data = preprocess()
        data.to_csv(out_path)

    '''filter admit=1'''
    train_data = data[data['admit'] == 1]

    '''group data accroding to user and random sample from each group'''
    def draw(deck, n):
        return deck.sample(n)
    train_data = train_data.groupby('userName').apply(draw, n=1)

    '''excluding train data, entire data are test data'''
    index = train_data.index.tolist()
    raw_index = [item[1] for item in index]
    test_data = data.drop(raw_index)

    train_data = train_data.drop(['Unnamed: 0'], 1)
    test_data = test_data.drop(['Unnamed: 0'], 1)

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    return train_data, test_data


def scoreConversion(feature, data, score_table):
    '''
    Utility function: Gre Old Score to New Score
    '''
    gre_score = list(data[feature])
    for i in range(len(gre_score)):
        if gre_score[i] > 170:
            try:
                if feature =='greV':
                    gre_score[i]=score_table['newV'][gre_score[i]]
                elif feature == 'greQ':
                    gre_score[i]=score_table['newQ'][gre_score[i]]
            except:
                continue
    return gre_score


if __name__ == '__main__':
    create_train_test()


