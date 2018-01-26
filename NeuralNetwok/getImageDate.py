# coding:utf8
import numpy as np
from PIL import Image
from sklearn.externals import joblib
import os


class ImageData:
    def __init__(self, image):
        self.image = image

    # 二值化
    def point(self, z=80):
        return self.image.point(lambda x: 1. if x > z else 0.)

    # 将二值化后的数组转化成网格特征统计图
    def get_features(self, imArray, num):
        # 拿到数组的高度和宽度
        h, w = imArray.shape
        data = []
        for x in range(0, w / num):
            offset_y = x * num
            temp = []
            for y in range(0, h / num):
                offset_x = y * num
                # 统计每个区域的1的值
                temp.append(sum(sum(imArray[0 + offset_y:num + offset_y, 0 + offset_x:num + offset_x])))
            data.append(temp)
        return np.asarray(data)

    def getData(self, num):
        img = self.point()
        # 将图片转换为数组形式，元素为其像素的亮度值
        img_array = np.asarray(img)
        # 得到网格特征统计图
        features_array = self.get_features(img_array, num)
        # print features_array
        return features_array.reshape(features_array.shape[0] * features_array.shape[1])