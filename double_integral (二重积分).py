import scipy.integrate
from numpy import exp
from math import sqrt
import math

# 创建表达式
f = lambda x,y : exp(x**2-y**2)

# 计算二重积分：（p:积分值，err:误差）
# 这里注意积分区间的顺序
# 第二重积分的区间参数要以函数的形式传入
p, err= scipy.integrate.dblquad(f, 0, 2, lambda g : 0, lambda h : 1)
print(p)

#使用nquad函数而不是dblquad
