import numpy as np


# 返回为包络直线: [k, b_upper, b_lower, b_upper - b_lower]
def solve_f_mz(x, y):
    result = []
    # 遍历 k
    for i, y_value in enumerate(y[:-1]):
        k_group = (y_value - y[i + 1:]) / (x[i] - x[i + 1:])
        for k in k_group:
            b_group = y - k * x
            b_upper = np.max(b_group)
            b_lower = np.min(b_group)
            result.append([k, b_upper, b_lower, b_upper - b_lower])
    # 最小区域筛选
    result = np.array(result)
    result = result[np.argmin(result[:, -1])]
    return result
