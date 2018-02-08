# -*-coding:utf-8 -*-
import numpy as np
import sklearn.decomposition

from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler
from scipy import io as spio
from matplotlib import pyplot as plt

'''
    StandardScaler作用:去均值和方差归一化
'''
data = spio.loadmat('data.mat')
X = data['X']
# print X
plt.scatter(X[:, 0], X[:, 1])
scaler = StandardScaler()
scaler.fit(X)
x_train = scaler.transform(X)

'''拟合数据'''
K = 1  # 要降的维度
model = pca.PCA(n_components=K).fit(x_train)  # 拟合数据，n_components定义要降的维度
print model.noise_variance_
print model.explained_variance_ratio_  # 返回 所保留的n个成分各自的方差百分比
print model.n_components_  # 返回所保留的成分个数n。
Z = model.transform(x_train)  # transform就会执行降维操作
# print Z


'''数据恢复并作图'''
Ureduce = model.components_  # 得到降维用的Ureduce
x_rec = np.dot(Z, Ureduce)  # 数据恢复
plt.scatter(x_rec[:, 0], x_rec[:, 1], c='r', marker='o')
plt.show()
