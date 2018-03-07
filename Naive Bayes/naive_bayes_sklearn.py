# -*-coding:utf-8-*-
import numpy as np
from sklearn.naive_bayes import GaussianNB
from matplotlib import pylab as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score

# 测试数据
iris = datasets.load_iris()
# print data

X = iris.data[:100, 1]
y = iris.data[:100, 2]
labels_train = iris.target[:100].reshape(-1)
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(X, y, c=labels_train)
plt.show()

# 实例化
clf = GaussianNB()
# 训练数据 fit相当于train
features_train = []
for i in range(100):
    features_train.append([X[i], y[i]])

features_train = np.array(features_train)
print features_train
print labels_train
clf.fit(features_train, labels_train)
# 输出单个预测结果
features_test = np.array([[2, 4], [3, 5], [4, 1], [3, 1]])
labels_test = np.array([1, 1, 0, 0])
# predict的参数要求是数组
pred = clf.predict(features_test)
print(pred)
# 准确度评估 评估正确/总数

# 方法1
accuracy = clf.score(features_test, labels_test)
print 'accuracy1=', accuracy

# 方法2
accuracy2 = accuracy_score(pred, labels_test)
print 'accuracy2=', accuracy2
