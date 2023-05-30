import numpy as np
from scipy import interpolate


def interpolate_func(x, y, x_grid, kind='cubic'):
    inter_func = interpolate.interp1d(x, y, kind=kind)
    y_grid = inter_func(x_grid)
    return np.array([x_grid, y_grid])


if __name__ == '__main__':
    pass
