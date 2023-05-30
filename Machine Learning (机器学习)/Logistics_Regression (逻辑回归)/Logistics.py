import numpy as np
import pandas as pd

from PSO import PSO


class Logistic_regression:
    def __init__(self, dataMat1, dataMat2):
        self.dataMat1 = np.array(dataMat1)  # 训练数据
        self.dataMat2 = np.array(dataMat2)  # 验证数据
        self.m1, self.n1 = self.dataMat1.shape  # 获取行数m1和列数n1
        self.beta = np.random.rand(self.n1) # 列为特征，最后一列为标签
    
    def aim_function(self, value):
        result = 0
        for unit in self.dataMat1:
            result = result + unit[-1] * np.log(1 / (1 + np.exp(-value[0][0] - np.dot(value[0][1:], unit[:-1].T)))) + (1 - unit[-1]) * np.log(1 - 1 / (1 + np.exp(-value[0][0] - np.dot(value[0][1:], unit[:-1].T))))
        return - result
    
    def Logistic_training(self):
        pso = PSO(self.aim_function, self.n1, 800, 500, 10, 20, 1e-4, C1=2, C2=2, W=1)  # 目标函数, 维度, 粒子个数, 迭代次数, 边界条件, 粒子最大速度, 截止条件
        fit_var_list, self.beta = pso.update_ndim()
        print("最优位置:" + str(self.beta))
        print("最优解:" + str(fit_var_list[-1]))
        return self.beta

    def Logistic_forecast(self):
        for unit in self.dataMat2:
            result = 1 / (1 + np.exp(-self.beta[0][0] - np.dot(self.beta[0][1:], unit.T)))
            if result >= 0.5:
                print("结果是:猪")
            elif result < 0.5:
                print("结果是:牛")

if __name__ == '__main__':
    data1 = pd.read_excel("good.xls", sheet_name=0, header=0,
                         index_col=0)  # "header:指定列名行，0取第一行，数据为列名行以下；index_col:指定列为索引列"
    data2 = pd.read_excel("unknown.xls", sheet_name=0, header=0,
                         index_col=0)  # "header:指定列名行，0取第一行，数据为列名行以下；index_col:指定列为索引列"
    logistic_training = Logistic_regression(data1, data2)
    logistic_training.Logistic_training()
    logistic_training.Logistic_forecast()
