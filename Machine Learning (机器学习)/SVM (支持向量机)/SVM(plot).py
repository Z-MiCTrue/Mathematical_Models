from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assumed you have, X (predictor数据) and Y (target标签) for training data set and X_test(predictor) of test_dataset
class SVM:
    def __init__(self, data1, data2, data, Kernel = 'rbf', C = 1):
        self.model = svm.SVC(kernel = Kernel, C= C, gamma='auto')
        self.X = data1
        self.Y = data2
        self.X_test = data
        self.SV_index = []

    def Training(self):
        self.model.fit(self.X, self.Y)
        self.SV_index = self.model.support_  # 支持向量索引
        return self.model.score(self.X, self.Y)

    def Plot_point(self):
        line_number = self.X.shape[0]
        if self.X.shape[1] >= 2:
            for i in range(line_number):
                if self.Y[i] == 1:
                    plt.scatter(self.X[i][0], self.X[i][1],c='b',s=20)
                else:
                    plt.scatter(self.X[i][0], self.X[i][1],c='y',s=20)
            for j in self.SV_index:
                plt.scatter(self.X[j][0], self.X[j][1], s=100, c = 'none', alpha=0.5, linewidth=1.5, edgecolor='red')
            plt.show()
        else:
            pass

    def Predicting(self):
        predicted = self.model.predict(self.X_test)
        print(predicted)

    # 调参:
    # sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)


if __name__ == '__main__':
    Tem_data1 = np.array(pd.read_excel("sample.xls", sheet_name=0, header=0,
                          index_col=0))  # "header:指定列名行，0取第一行，数据为列名行以下；index_col:指定列为索引列"
    Tem_data2 = np.array(pd.read_excel("unknown.xls", sheet_name=0, header=0,
                          index_col=0))  # "header:指定列名行，0取第一行，数据为列名行以下；index_col:指定列为索引列"
    Lable = np.array([1, 1, 1, -1, -1, -1])
    SVM_model = SVM(Tem_data1, Lable, Tem_data2, 'linear', 1)
    Score = SVM_model.Training()
    print('拟合度为:' + str(Score*100) + '%')
    SVM_model.Plot_point()
    SVM_model.Predicting()
