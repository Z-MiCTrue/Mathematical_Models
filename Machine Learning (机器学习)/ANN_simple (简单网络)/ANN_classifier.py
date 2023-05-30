import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np


class ANN_Net(nn.Module):
    def __init__(self, IO_num):
        super(ANN_Net, self).__init__()
        self.layer_h1 = nn.Linear(IO_num[0], 12)
        self.layer_h2 = nn.Linear(12, 6)
        self.layer_h3 = nn.Linear(6, 6)
        self.layer_h4 = nn.Linear(6, 6)
        self.layer_o = nn.Linear(6, IO_num[1])

    def forward(self, x_in):
        # In
        x_out = Func.rrelu(self.layer_h1(x_in), lower=1e-4, upper=1e-1)
        x_out = Func.elu(self.layer_h2(x_out))
        # Residuals 1
        x_out_ = Func.elu(self.layer_h3(x_out))
        x_out = x_out + x_out_
        # Residuals 2
        x_out_ = Func.elu(self.layer_h4(x_out))
        x_out = x_out + x_out_
        # Out
        x_out = Func.softmax(self.layer_o(x_out), dim=-1)
        return x_out


class My_ANN:
    def __init__(self, n_features, n_labels, lr=1e-3, weight_decay=1e-4):
        # 设备状态
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Device: ', self.device)
        # 构建数据集
        self.X_train = None
        self.Y_train = None
        self.X_pre = None
        self.label_list = None
        self.result_mat = None
        # 搭建网络
        self.ANN = ANN_Net((n_features, n_labels))
        self.ANN = self.ANN.to(self.device)
        self.n_features = n_features  # 用于检查网络
        self.n_labels = n_labels  # 用于检查网络
        # 设置优化器and损失函数
        self.optimizer = torch.optim.Adam(self.ANN.parameters(), lr=lr, weight_decay=weight_decay)  # L2正则化
        self.loss_func = nn.MSELoss()  # reduction 维度有无缩减,默认是'mean': 'none', 'mean', 'sum'

    def Data_import(self, X_train, Y_train, X_pre):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_pre = X_pre

    def Data_standardize(self):
        # 标准化数据集
        mean = np.average(self.X_train, axis=0)
        std = np.std(self.X_train, axis=0)
        self.X_train = (self.X_train - mean) / std
        self.X_pre = (self.X_pre - mean) / std
        # 变化数据集为张量
        self.X_train = torch.FloatTensor(self.X_train)
        self.Y_train = torch.FloatTensor(self.Y_train)
        if self.X_pre is not None:
            self.X_pre = torch.FloatTensor(self.X_pre)
        else:
            self.X_pre = None

    def M_train(self, t_times=1e4, max_loss=None):
        self.X_train = self.X_train.to(self.device)
        self.Y_train = self.Y_train.to(self.device)
        if max_loss is None:
            for epoch in range(int(t_times)):
                out = self.ANN(self.X_train)
                loss = self.loss_func(out, self.Y_train)  # 计算误差
                self.optimizer.zero_grad()  # 清除梯度
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新所有的参数
            return None
        else:
            times_overflow = True
            for epoch in range(int(t_times)):
                out = self.ANN(self.X_train)
                loss = self.loss_func(out, self.Y_train)  # 计算误差
                if loss.item() < max_loss:  # torch.item()  用于提取张量为浮点数
                    times_overflow = False
                    break
                else:
                    self.optimizer.zero_grad()  # 清除梯度
                    loss.backward()  # 反向传播
                    self.optimizer.step()  # 更新所有的参数
            # if times_overflow:
                # print('ANN Warning: Training Times Overflow')
            return times_overflow

    def M_predict(self):
        if self.X_pre is not None:
            self.X_pre = self.X_pre.to(self.device)
            self.result_mat = self.ANN(self.X_pre).to('cpu').data.numpy()  # torch.data 为一种深拷贝方法
        else:
            print('Data is None')

    def I_normalize(self, data_i):
        data_i = np.array(data_i)
        X_train = data_i[:, :-1]
        n_features = X_train.shape[1]
        label = data_i[:, -1]
        self.label_list = sorted(list(set(label)))
        n_labels = len(self.label_list)
        Y_train = np.zeros((len(data_i), n_labels), dtype=float)
        for i, element in enumerate(label):
            Y_train[i, self.label_list.index(element)] = 1
        # 分类数不一样时，重构网络
        if n_labels != self.n_labels:
            print('(Net Tips)-OUT change to: ', n_labels)
            self.n_labels = n_labels
            self.ANN = ANN_Net((self.n_features, self.n_labels))
            self.optimizer = torch.optim.Adam(self.ANN.parameters(), lr=1e-3, weight_decay=1e-4)
        # 特征数不一样时重构网络
        if n_features != self.n_features:
            print('(Net Tips)-IN change to: ', n_features)
            self.n_features = n_features
            self.ANN = ANN_Net((self.n_features, self.n_labels))
            self.optimizer = torch.optim.Adam(self.ANN.parameters(), lr=1e-3, weight_decay=1e-4)
        return X_train, Y_train

    def O_normalize(self):
        if self.result_mat is not None:
            label = np.argmax(self.result_mat)
            return self.label_list[label], self.result_mat  # 此处多训练时应考虑deepcopy方法
        else:
            print('Result is None')
            return None, None


if __name__ == '__main__':
    # 构建网络
    my_ANN = My_ANN(2, 4)
    # 构建输入集
    x = [[1, 1, 1],
         [-1, 1, 2],
         [-1, -1, 3],
         [1, -1, 4]]
    x_ = [1, 1]
    # 标准化数据集
    x, y = my_ANN.I_normalize(x)
    # 导入数据集
    my_ANN.Data_import(x, y, x_)
    # 正则化参数
    my_ANN.Data_standardize()
    # 训练数据集
    my_ANN.M_train(max_loss=1e-3)
    # 预测
    my_ANN.M_predict()
    # 输出
    # print(my_ANN.result_mat)
    print(my_ANN.O_normalize())
