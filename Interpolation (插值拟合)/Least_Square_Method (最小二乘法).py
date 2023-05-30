import numpy as np  # 科学计算库
import matplotlib.pyplot as plt  # 绘图库
from scipy.optimize import leastsq  # 引入最小二乘法算法


def aimFunc_poly(p, x):
    func = np.poly1d(p)
    return func(x)


class Least_Squares_fitter:
    def __init__(self, X, Y, aim_func, p_o):
        self.X = X
        self.Y = Y
        self.aim_func = aim_func
        self.p_o = p_o
        self.Para = None
        self.SSE = None

    def error_func(self, p, x, y):  # 偏差函数
        return self.aim_func(p, x) - y

    def LS_calculate(self):
        self.Para = leastsq(self.error_func, self.p_o, args=(self.X, self.Y))
        self.SSE = np.sum(np.power(self.aim_func(self.Para[0], self.X) - self.Y, 2))
        return self.Para[0], self.SSE

    def print_result(self):
        if self.Para is not None:
            print('Result is: ', self.Para[0])
            print('Cost is: ', self.Para[-1])
            print('SSE is: ', self.SSE)
        else:
            print('Error!')
            pass

    def plot_func(self, switch, x_range=None):  # 0, 化样本点; 1, 画函数拟合图像(x_range: [首, 末, 密度])
        if switch:  # 画样本点和画拟合直线
            if x_range is not None:
                x = np.linspace(x_range[0], x_range[1], x_range[-1])
            else:
                x = np.linspace(np.min(self.X), np.max(self.X))
            y = self.aim_func(self.Para[0], x)
            plt.figure()
            plt.scatter(self.X, self.Y, color="red", label="Samples", linewidth=2)
            plt.plot(x, y, color="blue", label="fitting", linewidth=2)
            plt.legend(loc='lower right')  # 绘制图例
            plt.show()
        else:  # 只画样本点
            plt.figure()
            plt.scatter(self.X, self.Y, color="red", label="Samples", linewidth=2)
            plt.show()


def fit_trends_cubic(price, days=np.array([0, 1, 2, 3, 4])):
    p_0 = np.random.rand(4)
    LS_f = Least_Squares_fitter(days, price, aimFunc_poly, p_0)
    p, SSE = LS_f.LS_calculate()
    LS_f.print_result()
    LS_f.plot_func(switch=True)  # check
    del LS_f
    return p, SSE


def fit_trends_liner(price, days=np.array([0, 1, 2, 3, 4])):
    p_0 = np.random.rand(2)
    LS_f = Least_Squares_fitter(days, price, aimFunc_poly, p_0)
    p, SSE = LS_f.LS_calculate()
    LS_f.print_result()
    LS_f.plot_func(switch=True)  # check
    del LS_f
    return p, SSE


if __name__ == '__main__':
    Yi = np.array([70.54922848782739,
                   78.07225225521299,
                   70.05913178388784,
                   64.25041140927871,
                   31.712956266315835,
                   31.712956266315835,
                   31.317773218139028,
                   79.79975113690753])
    fit_trends_cubic(Yi, np.arange(0, len(Yi)))
