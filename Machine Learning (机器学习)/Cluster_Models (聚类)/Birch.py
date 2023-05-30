import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.metrics import calinski_harabasz_score


def Draw_Sample(Sample_data):
    Sample_data = np.array(Sample_data)
    plt.scatter(Sample_data[:, 0], Sample_data[:, 1])
    plt.show()


def cluster_Birch(Sample_data, N_Clusters=None):
    Sample_data = np.array(Sample_data)
    # 参数说明: 
    # threshold: 叶节点每个CF的最大样本半径阈值T; 
    # branching_factor: CF-Tree内部节点的最大CF数B以及叶子节点的最大CF数L; 
    # n_clusters: 类别数K
    result_pre = Birch(n_clusters=N_Clusters, threshold=0.5, branching_factor=50).fit_predict(Sample_data)
    plt.scatter(Sample_data[:, 0], Sample_data[:, 1], c=result_pre)
    plt.show()
    # 当集群密集且分离良好时，C-H指数得分较高，这与集群的标准概念有关
    CH_Score = calinski_harabasz_score(Sample_data, result_pre)
    print(CH_Score)
    return result_pre, CH_Score
