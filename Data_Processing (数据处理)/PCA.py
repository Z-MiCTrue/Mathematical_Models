import numpy as np
from sklearn import decomposition,datasets


model = decomposition.PCA(n_components=2)
model.fit(X)
X_new = model.fit_transform(X)
Maxcomponent = model.components_
ratio = model.explained_variance_ratio_
score = model.score(X)
print('降维后的数据:',X_new)
print('返回具有最大方差的成分:',Maxcomponent)
print('保留主成分的方差贡献率:',ratio)
print('所有样本的log似然平均值:',score)
print('奇异值:',model.singular_values_)
print('噪声协方差:',model.noise_variance_)
g1=plt.figure(1,figsize=(8,6))
plt.scatter(X_new[:,0],X_new[:,1],c='r',cmap=plt.cm.Set1, edgecolor='k', s=40)
plt.xlabel('D1')
plt.ylabel('D2')
plt.title('After the dimension reduction')
plt.show()