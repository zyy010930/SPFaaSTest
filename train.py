'''train完整模块'''
import csv
import sys
# 用户：Ejemplarr
# 编写时间:2022/3/24 22:10
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from matplotlib import pyplot as plt
#from sympy import stats
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F #使用functional中的ReLu激活函数
import torch.optim as optim
from GRU import GRU
from TCN import TCN
from datapre import data_start, data_prediction_to_f_and_t, dataset_to_Dataset, dataset_split_4sets, lengths, targets, \
    data_prediction_zyy

'''
   数据的导入
   可调优数据的定义
   网络实例化
   优化器的定义
   数据搬移至gpu
   损失函数的定义
   开始训练
'''

# 可调参数的定义
BATCH_SIZE = 16
EPOCH = 10
LEARN_RATE = 1e-3

# 数据的导入
x, y = 0, 0
dataset_features, dataset_target = 0, 0
train_features, train_target, test_features, test_target = 0, 0, 0, 0
train_set = 0
train_set_iter = 0

pred = []

# 损失函数的定义
loss_fuc = nn.MSELoss()

predict1 = []
predict2 = []
predict3 = []

# 训练函数的定义
def train_for_gru(data, device, loss_fuc, net, optim, Epoch):
    for epoch in range(Epoch):
        loss_print = []
        for batch_idx, (x, y) in enumerate(data):
            x = x.reshape([BATCH_SIZE, lengths, 1])
            x = x.to(device)
            # print(y.shape)
            y = y.reshape((len(y), targets))
            y = y.to(device)
            # print(y.shape)
            y_pred = net(x)
            loss = loss_fuc(y, y_pred)
            loss_print.append(loss.item())
            # 三大步
            # 网络的梯度值更为0
            net.zero_grad()
            # loss反向传播
            loss.backward()
            # 优化器更新
            optim.step()
        #print('GRU:loss:', sum(loss_print) / len(data))



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GRU()
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            print("reset")
            layer.reset_parameters()
    net_gru = model.to(device)
    # 优化器的定义
    optim_gru = optim.Adam(params=net_gru.parameters(), lr=LEARN_RATE)
    start = time.perf_counter()
    train_for_gru(train_set_iter, device, loss_fuc, net_gru, optim_gru, EPOCH)
    end = time.perf_counter()
    print('训练时间为：{:.2f}s'.format(end - start))
    # 保存模型
    #torch.save(net_gru.state_dict(), 'gru.pt')

    # plt.plot(x, y)
    # plt.show()
    test_for_gru(test_features, net_gru, device)

def test_for_gru(dataset_features, net_gru, device):
    dataset_features = dataset_features.reshape([len(dataset_features), lengths, 1])
    #print(dataset_features)
    y_pred = net_gru(torch.from_numpy(dataset_features).to(device))
    y_pred = y_pred_to_numpy(y_pred)
    # y_pred = y_pred.reshape(y_pred.size/targets, targets)
    pred.extend(y_pred)
    print(len(pred))




def y_pred_to_numpy(y_pred):
    '''
    :param y_pred: 网络的输出
    :return: 一个numpy数组
    '''
    y_pred = y_pred.detach().cpu().numpy()
    return y_pred

path2 = './newData.csv'
df = pd.read_csv(path2, header=0)
df = df.iloc[:, :1440]


dt = None
#数据的准备
batch_size = 10

#CNN模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #两个卷积层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=3, padding=1)  #1为in_channels 10为out_channels
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3, padding=1)
        #池化层
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 3))  #2为分组大小2*2
        #全连接层 320 = 20 * 4 * 4
        self.fc = torch.nn.Linear(300, 3)

    def forward(self, x):
        #先从x数据维度中得到batch_size
        batch_size = x.size(0)
        #卷积层->池化层->激活函数
        # x = x.unsqueeze(1)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(batch_size, -1)  #将数据展开，为输入全连接层做准备
        x = self.fc(x)
        #print(x.shape)
        return x



model = TCN(input_size=15, output_size=1, num_channels=[32]*4, kernel_size=3, dropout=0.2)
# model = Net()
#在这里加入两行代码，将数据送入GPU中计算！！！
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)  #将模型的所有内容放入cuda中
print(torch.cuda.is_available())
print(device)
# #设置损失函数和优化器
# criterion = torch.nn.CrossEntropyLoss()
# #神经网络已经逐渐变大，需要设置冲量momentum=0.5
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

#训练
#将一次迭代封装入函数中
def train(epoch, X, target):
    running_loss = 0.0
    for index in range(200):
        inputs = X
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        targets1 = target.to(device, dtype=torch.float32).unsqueeze(-1)
        loss = criterion(outputs, targets1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if index == 199:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, index + 1, running_loss / 1000))

def test(index, X, target):
    with torch.no_grad():  #不需要计算梯度
        inputs = X
        targets1 = target
        inputs, targets1 = inputs.to(device), targets1.to(device)
        outputs = model(inputs)  # 得到预测输出
        print(outputs)
        predicted = outputs.data
        # _, targets1 = torch.max(targets1.data, dim=0)  # dim=1沿着索引为1的维度(行)
        num1 = 0
        num2 = 0
        num3 = 0
        predicted = predicted.cpu().numpy()
        out = []
        for i in range(len(predicted)):
            if predicted[i] <= 0.5:
                num1 += 1
                out.append(0)
            elif 1.5 > predicted[i] > 0.5:
                num2 += 1
                out.append(1)
            elif predicted[i] >= 1.5:
                num3 += 1
                out.append(2)
        print(num1)
        print(num2)
        print(num3)
        return out




if __name__ == '__main__':
    hit_rate = []
    TCN = []
    num = int(sys.argv[1])
    for i in range(num, num+1):
        print("第", i, "组:")
        x, y = data_start(i)
        for k in range(1, 4):
            for j in range(k):
                train_features, train_target = data_prediction_zyy(y[1440:], lengths, targets, k, j + 1)
                test_features, test_target = data_prediction_zyy(y[:1440], lengths, targets, k, j + 1)
                train_set = dataset_to_Dataset(data_features=train_features, data_target=train_target)

                train_set_iter = DataLoader(dataset=train_set,  # 将数据封装进Dataloader类
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,  # 打乱batch与batch之间的顺序
                                            drop_last=True)  # drop_last = True表示最后不够一个batch就舍弃那些多余的数据
                main()

            if k == 3:
                for m in range(464):
                    predict3.append(pred[m])
                    predict3.append(pred[465 + m])
                    predict3.append(pred[929 + m])
                predict3.append(pred[465])
                # predict3.append(pred[948])
                pred = []
            elif k == 2:
                for m in range(703):
                    predict2.append(pred[m])
                    predict2.append(pred[704 + m])
                predict2.append(pred[704])
                pred = []
            elif k == 1:
                for m in range(1440 - lengths - targets + 1):
                    predict1.append(pred[m])
                pred = []

        true_list = df.iloc[i, :]
        true_index = []
        num1 = 0
        num2 = 0
        num3 = 0
        for p in range(43, 1436):
            # true_index.append([0, 0, 0])
            m = sys.maxsize
            j0 = -1
            for j in range(3):
                sub = 0
                for k in range(5):
                    if j == 0:
                        sub += abs(true_list[p + k] - predict1[p-15][k])
                    elif j == 1:
                        sub += abs(true_list[p + k] - predict2[p - 29][k])
                    elif j == 2:
                        sub += abs(true_list[p + k] - predict3[p - 43][k])
                if m > sub:
                    m = sub
                    j0 = j
            # true_index[p-15][j0] = 1
            if j0 == 0:
                num1 += 1
            elif j0 == 1:
                num2 += 1
            elif j0 == 2:
                num3 += 1
            true_index.append(j0)
        print(num1)
        print(num2)
        print(num3)

        target = torch.tensor(np.array(true_index)).squeeze(-1)
        predict = []
        for i in range(1393):
            l = []
            for j in range(5):
                l.append(predict1[i+28][j])
                l.append(predict2[i+14][j])
                l.append(predict3[i][j])
            predict.append(l)
        predict = torch.tensor(predict)
        print(predict.shape)

        train(0, predict, target)
        out = test(i, predict, target)
        TCNpre = []
        for i in range(len(out)):
            if out[i] == 0:
                TCNpre.extend(predict1[i+28])
            elif out[i] == 1:
                TCNpre.extend(predict2[i+14])
            elif out[i] == 2:
                TCNpre.extend(predict3[i])
        TCN.append(TCNpre)

    with open('SPFaaS_pre.csv', 'a+', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the data
        writer.writerow(TCN[0])


