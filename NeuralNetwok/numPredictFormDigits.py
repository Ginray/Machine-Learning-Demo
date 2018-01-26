# -*-coding:utf-8 -*-
from NeuralNetwork import NeuralNetwork
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np


#加载数据集
digits = load_digits()
#训练集
X = digits.data
#标记
Y = digits.target
#数据与处理，让特征值都处在0-1之间
X -= X.min()
X /= X.max()


#切分训练集和测试集
'''
random_state:伪随机数生成器
'''
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#对标记进行二值化,比如0000000000代表数字0, 0100000000代表数组1, 0010000000代表数字2，依次化为该形式
labels_train = LabelBinarizer().fit_transform(y_train)


###########构造神经网络模型################
#构建神经网络结构
nn = NeuralNetwork([64, 100, 10], 'logistic')
#训练模型
nn.fit(X_train, labels_train, learning_rate=0.2, epochs=100)
#保存模型
# joblib.dump(nn, 'model/nnModel.m')
#加载模型
# nn = joblib.load('model/nnModel.m')


###############数字识别####################
#存储预测结果
predictions = []
#对测试集进行预测
for i in range(y_test.shape[0]):
    out = nn.predict(X_test[i])
    predictions.append(np.argmax(out))


###############模型评估#####################
#打印预测报告
print confusion_matrix(y_test, predictions)
#打印预测结果混淆矩阵
'''
准确率： 所有识别为”1”的数据中，正确的比率是多少。 
如识别出来100个结果是“1”， 而只有90个结果正确，有10个实现是非“1”的数据。 所以准确率就为90%

召回率： 所有样本为1的数据中，最后真正识别出1的比率。 
如100个样本”1”, 只识别出了93个是“1”， 其它7个是识别成了其它数据。 所以召回率是93%

F1-score:  是准确率与召回率的综合。 可以认为是平均效果。
'''
print classification_report(y_test, predictions)