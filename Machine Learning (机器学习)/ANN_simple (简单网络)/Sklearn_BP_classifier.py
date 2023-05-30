import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier

'''
参数:
hidden_​​layer_sizes: 第i个元素表示第i个隐藏层中的神经元数量
activation: identity 无操作激活, 返回f(x)= x
            logistic sigmoid函数,返回f(x)= 1 /(1 + exp(-x))
            tanh 双曲tan函数,返回f(x)= tanh(x)
            relu 整流后的线性单位函数,返回f(x)= max(0，x)
            softmax 多分类
slover: lbfgs 是准牛顿方法族的优化器
        sgd 指的是随机梯度下降
        adam 是指基于随机梯度的优化器
alpha: 及L2惩罚（正则化项）参数, float, 可选, 默认为0.0001
learning_rate: --
max_iter: 最大迭代次数, 对于随机解算器（'sgd'，'adam'）, 这决定了时期的数量(每个数据点的使用次数), 而不是梯度步数
random_state: int, 则random_state是随机数生成器使用的种子
              RandomState实例, 则random_state是随机数生成器
              None, 则随机数生成器是np.random使用的RandomState实例
tol: 默认1e-4, 优化的容忍度

函数:
fit(X, Y)
get_params([deep]) 获取此估算器的参数
predict(X) 使用多层感知器分类器进行预测
predict_proba(X) 概率估计
'''


class BP_ANN:
    def __init__(self, X_train, Y_train, X_pre):
        self.X_train = np.array(X_train)
        self.Y_train = np.array(Y_train)
        self.X_pre = np.array(X_pre)
        self.model = MLPClassifier(hidden_layer_sizes=(5,),
                                   activation='logistic',
                                   solver='adam',
                                   alpha=0.1,
                                   max_iter=10000,
                                   random_state=None,
                                   tol=1e-4)
        self.model.fit(self.X_train, self.Y_train)  # 训练模型

    def calculate_result(self, print_switch=0):
        pre = self.model.predict(self.X_pre)  # 模型预测
        pro = self.model.predict_proba(self.X_pre)  # 模型各概率预测
        if print_switch:
            print('预测标签为:', pre)
            print('预测概率矩阵为:\n', pro)
        return pre, pro

    def model_save(self):
        joblib.dump(self.model, 'BP_ANN result data.pkl')


if __name__ == '__main__':
    x_train = [[1, 1],
               [-1, 1],
               [-1, -1],
               [1, -1]]
    y_train = [1, 2, 3, 4]
    x_pre = [[4, 4],
             [-4, -4]]
    ANN_model_001 = BP_ANN(x_train, y_train, x_pre)
    ANN_model_001.calculate_result(1)
    del ANN_model_001
