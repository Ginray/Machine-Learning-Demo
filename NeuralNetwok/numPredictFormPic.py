# -*-coding:utf-8 -*-
from sklearn.externals import joblib
from NeuralNetwork import NeuralNetwork
from getImageDate import ImageData
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
from PIL import Image
###############提取图片中的特征向量####################
X = []
Y = []

for i in range(0, 10):
    # 遍历文件夹，读取数字图片
    for f in os.listdir("numImage/{0}".format(i)):
        # 打开一张文件并灰度化
        im = Image.open("numImage/%s/%s" % (i, f)).convert("L")
        # 使用ImageData类
        z = ImageData(im)
        # 获取图片网格特征向量，2代表每上下2格和左右两格为一组
        data = z.getData(2)
        X.append(data * 0.1)
        Y.append(i)


#切分训练集和测试集
'''
random_state:伪随机数生成器
'''
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#对标记进行二值化,比如0000000000代表数字0, 0100000000代表数组1, 0010000000代表数字2，依次化为该形式
labels_train = LabelBinarizer().fit_transform(y_train)


###########构造神经网络模型################
#构建神经网络结构
#因为构造出的图片是14*14的 所以NeuralNetwork的第一个参数是14*14 不然矩阵相乘会错误
nn = NeuralNetwork([14*14, 100,40, 10], 'logistic')
#训练模型
nn.fit(X_train, labels_train, learning_rate=0.2, epochs=100)
#保存模型
joblib.dump(nn, 'model/nnModel.m')
#加载模型
# nn = joblib.load('model/nnModel.m')


###############数字识别####################
#存储预测结果
predictions = []
#对测试集进行预测
y_test = np.array(y_test)
for i in range(y_test.shape[0]):
    out = nn.predict(X_test[i])
    '''
    numpy.argmax()返回沿轴axis最大值的索引。
    '''
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