import numpy as np
import cv2


class Optimizer_adam:
    def __init__(self, variable_len, lr=1e0):
        self.lr = lr
        self.eps = 1e-9
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.integrate = np.array([1e0] * variable_len)
        self.variable_range = np.array([[-np.inf, np.inf]] * variable_len)
        self.vector_x = np.array([None] * variable_len)
        self.grad = np.zeros(variable_len)
        self.grad_smooth = np.zeros(variable_len)
        self.velocity_smooth = np.zeros(variable_len)
        self.variable_len = variable_len
        self.used_map = cv2.imread('test.jpg', 0)

    def aim_function(self, point_t):
        return -self.used_map[point_t[0], point_t[1]]

    def run(self, max_loss=1e-4):
        # Initialize starting point (omitted)
        # start iterating
        for i in range(1, 100):
            # compute gradient
            for j in range(self.variable_len):
                temp_x = self.vector_x.copy()
                temp_x[j] += self.integrate[j]
                self.grad[j] = (self.aim_function(temp_x) - self.aim_function(self.vector_x)) / self.integrate[j]
            # Exponential smoothing calculation
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
                    if step[j] >= 1:
                        self.vector_x[j] += step[j]
            # Terminate when gradient disappears
            if np.mean(np.abs(self.grad)) <= max_loss:
                # print('warning! max loss reached: <grade: ', self.grad, '>')
                break
        return self.vector_x


if __name__ == '__main__':
    Oa = Optimizer_adam(2)
    Oa.vector_x = np.array([3, 3])
    print(Oa.run())
    pass

'''
developed by Z
'''