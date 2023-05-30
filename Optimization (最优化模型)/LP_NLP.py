import numpy as np
from scipy.optimize import minimize


def constraint(x_0):
    # 约束条件 分为eq 和ineq (eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0)
    # 总边界条件
    cons = [{'type': 'ineq', 'fun': lambda x: 48000 - x[0] - x[1] - x[2]},
            {'type': 'ineq', 'fun': lambda x: x[0] / 0.6 + x[1] / 0.66 + x[2] / 0.72 - 28200},
            {'type': 'ineq', 'fun': lambda x: x[:]}
            ]
    return cons


def aim_fun():
    min_fun = lambda x: 10 * (1.2 * x[0] + 1.1 * x[1] + x[2]) + (x[0] + x[1] + x[2])
    return min_fun


def LP_calculation():
    x_0 = np.array([1000, 1000, 1000])  # 初始值
    cons = constraint(x_0)
    res = minimize(aim_fun(), x_0, method='SLSQP', constraints=cons)  # SLSQP;
    print('目标函数结果: ' + str(res.fun))
    print(res.success)
    print(res.x)
    return res.x, res.success


if __name__ == '__main__':
    LP_calculation()
