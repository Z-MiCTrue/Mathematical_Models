import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn import metrics

'''
# 参数说明
eps          # 邻域半径, 默认值 = 0.5
min_samples  # 邻域样本最小数量阈值, 默认值 = 5
metric       # 样本距离计算标准, 取默认'euclidean'即欧几里得距离
algorithm    # 样本距离计算算法, 取默认'auto', 可能是'brute'&'ball_tree'&'kd_tree'
leaf_size    # 叶子大小传递给BallTree或cKDTree, 默认值 = 30
'''


def Draw_Sample(Sample_data):
    Sample_data = np.array(Sample_data)
    plt.scatter(Sample_data[:, 0], Sample_data[:, 1])
    plt.show()


def cluster_Birch(Sample_data):
    Sample_data = np.array(Sample_data)
    result_pre = DBSCAN(eps=0.14,
                        min_samples=2,
                        metric='euclidean',
                        metric_params=None,
                        algorithm='auto',
                        leaf_size=30).fit_predict(Sample_data)
    plt.scatter(Sample_data[:, 0], Sample_data[:, 1], c=result_pre)
    plt.show()
    # 当集群密集且分离良好时，Calinski-Harabasz指数得分较高，这与集群的标准概念有关
    CH_Score = metrics.calinski_harabasz_score(Sample_data, result_pre)
    print(CH_Score)
    return result_pre, CH_Score


if __name__ == '__main__':
    X, Y=datasets.make_circles(n_samples=1000, factor=.6, noise=.05)
    y_pred = DBSCAN(eps=0.14,
                    min_samples=2,
                    metric='euclidean',
                    metric_params=None,
                    algorithm='auto',
                    leaf_size=30).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.show()