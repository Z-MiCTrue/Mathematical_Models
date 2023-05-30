import numpy as np
from sklearn.impute import KNNImputer

def KNN_replenishment(dataMat, marker):  # marker 为缺失值标记['No', ...]
    dataMat = np.array(dataMat)
    for i in range(len(dataMat)):
        for j in range(len(dataMat)):
            if unit in marker:
                dataMat[i][j] = np.nan
    imputer = KNNImputer(n_neighbors=1)
    dataMat = imputer.fit_transform(dataMat)
    return dataMat