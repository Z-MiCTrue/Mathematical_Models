import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np


class ANN_Net(nn.Module):
    def __init__(self, IO_num):
        super(ANN_Net, self).__init__()
        # In
        self.layer_in = nn.Linear(IO_num, 8)
        self.layer_h1 = nn.Linear(8, 4)
        # Residuals 1
        self.layer_h2 = nn.Linear(4, 4)
        self.layer_h3 = nn.Linear(4, 4)
        # Residuals 2
        self.layer_h4 = nn.Linear(4, 4)
        self.layer_h5 = nn.Linear(4, 4)
        # encode
        self.layer_encode = nn.Linear(4, 1)
        # decode
        self.layer_h6 = nn.Linear(1, 4)
        # Residuals 3
        self.layer_h7 = nn.Linear(4, 4)
        self.layer_h8 = nn.Linear(4, 4)
        # Residuals 4
        self.layer_h9 = nn.Linear(4, 4)
        self.layer_h10 = nn.Linear(4, 4)
        # Out
        self.layer_decode = nn.Linear(4, IO_num)
        self.data_mean = 0
        self.data_std = 1

    def forward(self, x_in):
        x_in = (x_in - self.data_mean) / self.data_std
        # In
        x_out = self.layer_h1(Func.rrelu(self.layer_in(x_in), lower=1e-3, upper=4e-3))
        # Residuals 1
        x_out_ = self.layer_h3(Func.rrelu(self.layer_h2(x_out), lower=1e-3, upper=4e-3))
        x_out = x_out + x_out_
        # Residuals 2
        x_out_ = self.layer_h5(Func.rrelu(self.layer_h4(x_out), lower=1e-3, upper=4e-3))
        x_out = x_out + x_out_
        # encode
        x_encode = Func.rrelu(self.layer_encode(x_out), lower=1e-3, upper=4e-3)
        # decode
        x_out = Func.rrelu(self.layer_h6(x_encode), lower=1e-3, upper=4e-3)
        # Residuals 3
        x_out_ = self.layer_h8(Func.rrelu(self.layer_h7(x_out), lower=1e-3, upper=4e-3))
        x_out = x_out + x_out_
        # Residuals 4
        x_out_ = self.layer_h10(Func.rrelu(self.layer_h9(x_out), lower=1e-3, upper=4e-3))
        x_out = x_out + x_out_
        # Out
        x_decode = Func.rrelu(self.layer_decode(x_out), lower=1e-3, upper=4e-3)
        return x_encode, x_decode


class My_ANN:
    def __init__(self, n_features, lr=1e-3, weight_decay=1e-4):
        # 设备状态
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Device: ', self.device)
        # 构建数据集
        self.X_train = None
        # 搭建网络
        self.ANN = ANN_Net(n_features)
        self.ANN = self.ANN.to(self.device)
        # 保存相关参数
        self.n_features = n_features  # 用于检查网络
        self.lr = lr  # 用于检查网络
        self.weight_decay = weight_decay  # 用于检查网络
        # 设置优化器and损失函数
        self.optimizer = torch.optim.Adam(self.ANN.parameters(), lr=lr, weight_decay=weight_decay)  # L2正则化
        self.loss_func = nn.MSELoss()  # reduction 维度有无缩减, 默认是 mean: 'none', 'mean', 'sum'

    def M_train(self, t_times=1e4, max_loss=0.):
        # 求解正则化参数
        self.ANN.data_mean = torch.FloatTensor(np.average(self.X_train, axis=0)).to(self.device)
        self.ANN.data_std = torch.FloatTensor(np.std(self.X_train, axis=0)).to(self.device)
        # 变化数据集从numpy数组为张量
        if type(self.X_train) is np.ndarray:
            self.X_train = torch.from_numpy(self.X_train).float()
        elif type(self.X_train) is torch.Tensor:
            pass
        else:
            self.X_train = torch.FloatTensor(self.X_train)
        # 将数据集转移至目设备
        self.X_train = self.X_train.to(self.device)
        # 开始迭代
        times_overflow = True
        for epoch in range(int(t_times)):
            x_encode, x_decode = self.ANN(self.X_train)
            loss = self.loss_func(x_decode, self.X_train)  # 计算误差
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
        self.X_train = data_in

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
        x_encode, x_decode = self.ANN(X_pre)
        return x_encode.to('cpu').data.numpy(), x_decode.to('cpu').data.numpy()

    def I_normalize(self, data_i):
        X_train = np.array(data_i)
        n_features = X_train.shape[1]
        # 特征数不一样时重构网络
        if n_features != self.n_features:
            print('(Net Tips)-IN change to: ', n_features)
            self.n_features = n_features
            self.ANN = ANN_Net(self.n_features)
            self.optimizer = torch.optim.Adam(self.ANN.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return X_train


if __name__ == '__main__':
    data = np.array([[9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                     [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                     [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
                     [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    my_ANN = My_ANN(6, lr=1e-3, weight_decay=1e-4)
    # 导入数据集
    my_ANN.Data_import(my_ANN.I_normalize(data))
    # 训练数据集
    my_ANN.M_train(max_loss=1e-3)
    print(my_ANN.Net_forward(data))
    pass
