import matplotlib.pyplot as plt
import numpy as np


def sl_plot(y, x=None,  save_switch=False):
    if x is None:
        x = np.arange(0, len(y))
    else:
        x = np.array(x)
    y = np.array(y)
    plt.figure()
    plt.plot(x, y, linewidth=2, color='b', marker='o', markerfacecolor='black', markersize=2)
    plt.title('Trends')
    plt.ylabel('Value')
    plt.xlabel('Variable')
    # plt.legend()
    if save_switch:
        plt.savefig('Statistic.png')
    else:
        pass
    plt.show()
