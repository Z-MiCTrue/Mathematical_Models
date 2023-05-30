import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def aim_function_1(x, y):
    z = np.sin(x) + np.sin(y)
    return z


def aim_function_2(x, y):
    z = x + y
    return z


class function_plot_3D:
    def __init__(self, function_list, x_range, y_range, interval=0.01):
        fig = plt.figure()
        self.ax = fig.add_axes(Axes3D(fig))
        x = np.arange(x_range[0], x_range[-1], interval)
        y = np.arange(y_range[0], y_range[-1], interval)
        self.X, self.Y = np.meshgrid(x, y)  # 网格的创建
        self.Z = []
        for func in function_list:
            self.Z.append(func(self.X, self.Y))
        self.x_label = 'x'
        self.y_label = 'y'
        self.fig_name = 'Function_image.png'

    def para_adjust(self, x_label, y_label, fig_name):
        self.x_label = x_label
        self.y_label = y_label
        self.fig_name = fig_name

    def f_plot(self, save_switch):
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        for value_z in self.Z:
            self.ax.plot_surface(self.X, self.Y, value_z, rstride=1, cstride=1, cmap='rainbow')
        if save_switch:
            plt.savefig(self.fig_name)
        plt.show()


if __name__ == '__main__':
    f_image_001 = function_plot_3D([aim_function_1, aim_function_2], [-10, 10], [-10, 10], interval=0.1)
    f_image_001.f_plot(0)
