#-*-coding:utf-8-*-

#测试数据
import numpy as np
from sklearn.naive_bayes import GaussianNB
from matplotlib import pylab as plt

features_train = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
labels_train = np.array([1, 1, 1, 2, 2, 2])

X = features_train[:,1]
y = features_train[:,0]
plt.scatter(X,y)
plt.show()

#实例化
clf = GaussianNB()
#训练数据 fit相当于train
clf.fit(features_train, labels_train)
#输出单个预测结果
features_test = np.array([[-0.8,-1],[-1,-3],[2,1],[3,4]])
labels_test = np.array([1,1,2,2])
#predict的参数要求是数组
pred = clf.predict(features_test)
print(pred)
#准确度评估 评估正确/总数
#方法1
accuracy = clf.score(features_test, labels_test)
print 'accuracy1=',accuracy
#方法2
from sklearn.metrics import accuracy_score
accuracy2 = accuracy_score(pred,labels_test)
print 'accuracy2=',accuracy2