import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np


class ANN_Net(nn.Module):
    def __init__(self, IO_num):
        super(ANN_Net, self).__init__()
        self.layer_h1 = nn.Linear(IO_num[0], 8)
        self.layer_h2 = nn.Linear(8, 6)
        self.layer_h3 = nn.Linear(6, 6)
        self.layer_h4 = nn.Linear(6, 6)
        self.layer_o = nn.Linear(6, IO_num[1])
        self.data_mean = 0
        self.data_std = 1

    def forward(self, x_in):
        x_in = (x_in - self.data_mean) / self.data_std
        # In
        x_out = Func.rrelu(self.layer_h1(x_in), lower=1e-3, upper=4e-3)
        x_out = Func.rrelu(self.layer_h2(x_out), lower=1e-3, upper=4e-3)
        # Residuals 1
        x_out_ = Func.rrelu(self.layer_h3(x_out), lower=1e-3, upper=4e-3)
        x_out = x_out + x_out_
        # Residuals 2
        x_out_ = Func.rrelu(self.layer_h4(x_out), lower=1e-3, upper=4e-3)
        x_out = x_out + x_out_
        # Out
        x_out = torch.tanh(self.layer_o(x_out))
        return x_out


class My_ANN:
    def __init__(self, n_features, n_out, lr=1e-3, weight_decay=1e-4):
        # 设备状态
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Device: ', self.device)
        # 构建数据集
        self.X_train = None
        self.Y_train = None
        # 搭建网络
        self.ANN = ANN_Net((n_features, n_out))
        self.ANN = self.ANN.to(self.device)
        # 保存相关参数
        self.n_features = n_features  # 用于检查网络
        self.n_out = n_out  # 用于检查网络
        self.lr = lr  # 用于检查网络
        self.weight_decay = weight_decay  # 用于检查网络
        # 设置优化器and损失函数
        self.optimizer = torch.optim.Adam(self.ANN.parameters(), lr=lr, weight_decay=weight_decay)  # L2正则化
        self.loss_func = nn.MSELoss()  # reduction 维度有无缩减,默认是'mean': 'none', 'mean', 'sum'

    def M_train(self, t_times=1e4, max_loss=0.):
        # 求解正则化参数
        self.ANN.data_mean = torch.FloatTensor(np.average(self.X_train, axis=0)).to(self.device)
        self.ANN.data_std = torch.FloatTensor(np.std(self.X_train, axis=0)).to(self.device)
        # 变化数据集从numpy数组为张量
        if type(self.X_train) is np.ndarray and type(self.Y_train) is np.ndarray:
            self.X_train = torch.from_numpy(self.X_train).float()
            self.Y_train = torch.from_numpy(self.Y_train).float()
        elif type(self.X_train) is torch.Tensor and type(self.Y_train) is torch.Tensor:
            pass
        else:
            self.X_train = torch.FloatTensor(self.X_train)
            self.Y_train = torch.FloatTensor(self.Y_train)
        # 将数据集转移至目设备
        self.X_train = self.X_train.to(self.device)
        self.Y_train = self.Y_train.to(self.device)
        # 开始迭代
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
        if times_overflow:
            print('ANN Warning: Training Times Overflow')
        return times_overflow

    def Data_import(self, data_in):
        self.X_train, self.Y_train = data_in

    def Net_forward(self, X_pre):
        # 变化数据集从numpy数组为张量
        if type(X_pre) is np.ndarray:
            X_pre = torch.from_numpy(X_pre).float()
        elif type(X_pre) is torch.Tensor:
            pass
        else:
            X_pre = torch.FloatTensor(X_pre)
        # 将数据集转移至目设备
        X_pre = X_pre.to(self.device)
        X_out = self.ANN(X_pre).to('cpu').data.numpy()  # torch.data 为一种深拷贝方法
        return X_out

    def I_normalize(self, data_i):
        data_i = np.array(data_i)
        X_train = data_i[:, :-1]
        n_features = X_train.shape[1]
        Y_train = data_i[:, -1, None]
        # 特征数不一样时重构网络
        if n_features != self.n_features:
            print('(Net Tips)-IN change to: ', n_features)
            self.n_features = n_features
            self.ANN = ANN_Net((self.n_features, self.n_out))
            self.optimizer = torch.optim.Adam(self.ANN.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return X_train, Y_train


if __name__ == '__main__':
    data = np.array([[0, 0, 0],
                     [0.1, 0.2, 0.3],
                     [0.2, 0.1, 0.3],
                     [0.2, 0.2, 0.4],
                     [0.2, 0.3, 0.5],
                     [0.3, 0.2, 0.5]])
    x_pre = np.array([[0.3, 0.3]])
    my_ANN = My_ANN(2, 1, lr=1e-3, weight_decay=1e-4)
    # 导入数据集
    my_ANN.Data_import(my_ANN.I_normalize(data))
    # 训练数据集
    my_ANN.M_train(max_loss=1e-3)
    # 预测
    print(my_ANN.Net_forward(x_pre))
    pass
