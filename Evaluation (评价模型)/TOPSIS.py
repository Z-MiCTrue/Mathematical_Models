#coding=utf-8
import pandas as pd
import numpy as np

# 正向化函数-->极大(此时数据列表已化为array)
def dataDirection_1(datas):
    return np.max(datas) - datas

def dataDirection_2(datas, a, b):
    M = np.max([a-np.min(datas), np.max(datas)-b])
    for i in len(datas):
        if datas[i] < a:
            data[i] = 1 - (a - data[i])/M
        elif datas[i] > b:
            data[i] = 1 - (data[i] - b)/M
        else:
            data[i] = 1
    return data

# 标准化矩阵函数(消除量纲影响)
def Standardise(datas):
    return datas / np.sqrt(np.sum(datas**2, axis=0))  # axis=0表示列相加


# 熵权定权重
def E_j(dataMat):  #计算熵值
    dataMat = np.array(dataMat)
    E = np.array(None_ij)
    for i in range(m):
        for j in range(n):
            if dataMat[i][j] == 0:
                e_ij = 0.0
            else:
                P_ij = dataMat[i][j] / dataMat.sum(axis=0)[j]  #计算比重
                e_ij = (-1 / np.log(m)) * P_ij * np.log(P_ij)
            E[i][j] = e_ij
    E_j=E.sum(axis=0)
    E_j = E_j(Y_ij)  # 熵值
    G_j = 1 - E_j  # 计算差异系数
    W_j = G_j / sum(G_j)  # 计算权重
    return W_j


def TOPSIS_WSD(data_name, W_j): #"data_name: xls文件，且横为属性，纵为对象；"
    data = pd.read_excel(data_name, sheet_name=0, header=0, index_col=0) #"header:指定列名行，0取第一行，数据为列名行以下；index_col:指定列为索引列"
    Y_ij = np.array(data)
    #Y_ij[:,2] = dataDirection_1(Y_ij[:,2])  #正向化数据(Attention:数据此时已去分类栏)
    Y_ij = Standardise(Y_ij)  #标准化矩阵
    m, n = Y_ij.shape  # 获取行数m和列数n
    None_ij = [[None] * n for i in range(m)]
    # 决定权重并打印:
    W_j = np.array(W_j)
    # W_j = np.random.rand(n)   #随机权重
    # WW = pd.Series(W_j, index=data.columns, name='指标权重')
    # WW.to_excel("权重表.xls",sheet_name='WW')
    #TOPSIS计算:
    Z_ij = np.array(None_ij)  # 空矩阵
    for i in range(m):
        for j in range(n):
            Z_ij[i][j] = Y_ij[i][j] * np.sqrt(W_j[j])  # 计算加权标准化矩阵Z_ij
    Imax_j = Z_ij.max(axis=0)  # 最优解
    Imin_j = Z_ij.min(axis=0)  # 最劣解
    Dmax_ij = np.array(None_ij)
    Dmin_ij = np.array(None_ij)
    for i in range(m):
        for j in range(n):
            Dmax_ij[i][j] = (Imax_j[j] - Z_ij[i][j]) ** 2
            Dmin_ij[i][j] = (Imin_j[j] - Z_ij[i][j]) ** 2
    Dmax_i = Dmax_ij.sum(axis=1) ** 0.5  # 最优解欧氏距离
    Dmin_i = Dmin_ij.sum(axis=1) ** 0.5  # 最劣解欧氏距离
    C_i = Dmin_i / (Dmax_i + Dmin_i)  # 综合评价值
    Dmax_i = pd.Series(Dmax_i, index=data.index, name='最优解')
    Dmin_i = pd.Series(Dmin_i, index=data.index, name='最劣解')
    C_i = pd.Series(C_i, index=data.index, name='综合评价值')
    pd.concat([C_i]).to_excel("TOPTSIS_result.xls")