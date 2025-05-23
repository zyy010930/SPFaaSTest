# 用户：Cy
# 编写时间:2022/3/26 19:53

'''data_prediction完整模块'''

# 用户：Ejemplarr
# 编写时间:2022/3/24 22:11
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

'''
    lengths :决定了用于预测序列的长度
    targets :表示待预测的序列长度
    例如lengths = 8， targets = 1，则表示用8个数预测一个数
'''
lengths = 15
targets = 5

def data_start(nm):
    path = './newData.csv'
    dt = pd.read_csv(path, header=0)
    print(dt.shape)
    l = dt.iloc[nm, :2880]
    y = [0 for m in range(2880)]
    x = []
    n = [0.01, 0.05, 0.2, 0.5, 1]
    sum = 0
    for j in range(2880):
        x.append(j+1)
        if l[j] > 0:
            sum += 1
            for k in range(max(0, j - 4), min(2879, j + 5)):
                y[k] += l[j]*n[(5-abs(k-j))-1]

    y = torch.tensor(y, dtype=torch.float32)
    x = torch.tensor(x, dtype=torch.float32)

    return x, y

def data_prediction_to_f_and_t(data, num_features, num_targets):
    '''
    准备数据集的函数
    '''
    features, target = [], []

    for i in range(((len(data)-num_features-num_targets)//num_targets) + 1):
        f = data[i*num_targets:i*num_targets+num_features]
        t = data[i*num_targets+num_features:i*num_targets+num_features+num_targets]
        features.append(list(f))
        target.append(list(t))

    return np.array(features), np.array(target)

def data_prediction_zyy(data, num_features, num_targets, interval, num):
    '''
    准备数据集的函数
    '''
    features, target = [], []

    for i in range((len(data)-interval*(num_features-1)-1-num_targets) + 1):
        if (i+1) % interval == num % interval:
            f = data[i:i+(num_features-1)*interval+1:interval]
            t = data[i+(num_features-1)*interval+1:i+(num_features-1)*interval+1+num_targets]
            features.append(list(f))
            target.append(list(t))

    return np.array(features), np.array(target)

class dataset_to_Dataset(Dataset):
    '''
    将传入的数据集，转成Dataset类，方面后续转入Dataloader类
    注意定义时传入的data_features,data_target必须为numpy数组
    '''
    def __init__(self, data_features, data_target):
        self.len = len(data_features)
        self.features = torch.from_numpy(data_features)
        self.target = torch.from_numpy(data_target)

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len

def dataset_split_4sets(data_features, data_target, ratio=0.5):
    '''
    功能：训练集与测试集的特征与target分离
    ratio:表示训练集所占的百分比
    '''
    split_index = int(ratio*len(data_features))
    train_features = data_features[:split_index]
    train_target = data_target[:split_index]
    test_features = data_features[split_index:]
    test_target = data_target[split_index:]
    return train_features, train_target, test_features, test_target