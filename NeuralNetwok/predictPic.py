# -*-coding:utf-8 -*-
from sklearn.externals import joblib
from getImageDate import ImageData
from PIL import Image
import numpy as np

# image = input('input image path\n')
image = r'C:\Users\ciabok\Desktop\2.bmp'
# image = raw_input('please input image path\n')

im = Image.open(image).convert("L")
im = im.resize((28,28))
# im.show()
z = ImageData(im)
# 获取图片网格特征向量，2代表每上下2格和左右两格为一组
data = z.getData(2) * 0.1
print data
nn = joblib.load('model/nnModel.m')
out = nn.predict(data)
print out
result = np.argmax(out)

print 'result = ',result
