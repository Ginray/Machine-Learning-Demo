# -*-coding:utf-8 -*-
from sklearn.externals import joblib
from getImageDate import ImageData
from PIL import Image
import numpy as np

# image = input('input image path\n')
image = r'C:\Users\ciabok\Desktop\1.bmp'
im = Image.open(image).convert("L")
z = ImageData(im)
# 获取图片网格特征向量，2代表每上下2格和左右两格为一组
data = z.getData(2) * 0.1

nn = joblib.load('model/nnModel.m')
out = nn.predict(data)
print out
result = np.argmax(out)

print result
