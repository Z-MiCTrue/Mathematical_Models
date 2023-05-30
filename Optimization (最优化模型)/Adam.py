from copy import deepcopy

import numpy as np


def test_func(x):
    return (x[0] - 10) ** 2 + (x[1] - 20) ** 2


class Optimizer_adam:
    def __init__(self, variable_len, lr=1e-3, integrate=1e-4):
        self.lr = lr
        self.eps = 1e-9
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        # auto
        self.aim_function = None
        self.integrate = np.array([integrate] * variable_len)
        self.variable_range = np.array([[-np.inf, np.inf]] * variable_len)
        self.vector_x = np.array([None] * variable_len)
        self.grad = np.zeros(variable_len)
        self.grad_smooth = np.zeros(variable_len)
        self.velocity_smooth = np.zeros(variable_len)
        self.variable_len = variable_len
        self.min_loss = np.inf
        self.best_solve = None

    def reinitialize(self):
        self.grad = np.zeros(self.variable_len)
        self.grad_smooth = np.zeros(self.variable_len)
        self.velocity_smooth = np.zeros(self.variable_len)
        self.variable_len = self.variable_len
        self.min_loss = np.inf
        self.best_solve = None

    def run(self, max_time=1e4, min_grade=1e-4):
        # 初始化起点
        if None in self.vector_x:
            for i in range(self.variable_len):
                if self.variable_range[i].all() == np.array([-np.inf, np.inf]).all():
                    self.vector_x[i] = 0
                elif self.variable_range[i, 0] == -np.inf:
                    self.vector_x[i] = self.variable_range[i, 1] - 1e-4
                elif self.variable_range[i, 1] == np.inf:
                    self.vector_x[i] = self.variable_range[i, 0] + 1e-4
                else:
                    self.vector_x[i] = np.random.randint(self.variable_range[i, 0], self.variable_range[i, 1])
        for i in range(1, int(max_time + 1)):
            # 计算 loss
            loss = self.aim_function(self.vector_x)
            # 计算梯度
            for j in range(self.variable_len):
                temp_x = deepcopy(self.vector_x)
                temp_x[j] += self.integrate[j]
                self.grad[j] = (self.aim_function(temp_x) - loss) / self.integrate[j]
            # 指数平滑计算
            self.grad_smooth = self.beta_1 * self.grad_smooth + (1 - self.beta_1) * self.grad
            self.velocity_smooth = self.beta_2 * self.velocity_smooth + (1 - self.beta_2) * np.power(self.grad, 2)
            step = -(self.lr * self.grad_smooth) / (np.power(self.velocity_smooth, 1 / 2) + self.eps)
            for j in range(self.variable_len):
                if self.vector_x[j] + step[j] > self.variable_range[j, 1]:
                    self.vector_x[j] = self.variable_range[j, 1]
                    self.grad[j] = 0
                elif self.vector_x[j] + step[j] < self.variable_range[j, 0]:
                    self.vector_x[j] = self.variable_range[j, 0]
                    self.grad[j] = 0
                else:
                    self.vector_x[j] += step[j]
            # 更新历史最佳结果
            if loss < self.min_loss:
                self.min_loss = loss
                self.best_solve = deepcopy(self.vector_x)
            # 梯度消失时终止
            if np.max(np.abs(self.grad)) <= min_grade:
                print(f'warning! max loss reached: <grade: {self.grad}>')
                break
            # 定时输出
            if i % 1000 == 0:
                print('vector_x= ', self.vector_x, '; result= ', self.aim_function(self.vector_x))
        return deepcopy(self.best_solve)


if __name__ == '__main__':
    Oa = Optimizer_adam(2, lr=1e-3, integrate=1e-4)
    Oa.vector_x = np.array([7., 17.])
    Oa.aim_function = test_func
    solve = Oa.run(max_time=1e4, min_grade=1e-4)
    print(solve)
    pass
