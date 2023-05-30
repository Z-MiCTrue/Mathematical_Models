import numpy as np
from matplotlib import pyplot as plt


def aim_funcion(x):
    si = 12
    mu = 50
    y = (1 / (np.sqrt(2 * np.pi) * si)) * np.exp(-(x - mu) ** 2 / (2 * (si ** 2)))
    return y


class function_plot():
    def __init__(self, aim_funcion, x_min, x_max, interval = 0.001):
        self.x = np.arange(x_min, x_max, interval)
        self.y = aim_funcion(self.x)
        self.x_label = 'x'
        self.y_label = 'y'
        self.fig_title = 'Function_image'
        self.fig_name = 'Function_image.png'
        
    def para_adjust(self, x_label, y_label, fig_title, fig_name):
        self.x_label = x_label
        self.y_label = y_label
        self.fig_title = fig_title
        self.fig_name = fig_name
    
    def f_plot(self, save_switch):
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.fig_title)
        plt.plot(self.x, self.y)
        if save_switch:
            plt.savefig(self.fig_name)
        plt.show()

if __name__ == '__main__':
    f_image_001 = function_plot(aim_funcion, 0, 100)
    f_image_001.f_plot(0)
