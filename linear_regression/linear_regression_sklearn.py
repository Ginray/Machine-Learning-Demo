# -*-coding:utf-8 -*-

from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
X = np.random.random(size=(20, 1))
y = 4 * X.squeeze() + np.random.random(20)

plt.scatter(X, y)

model = linear_model.LinearRegression()
model.fit(X, y)
print model.coef_, model.intercept_
result = model.coef_ * X + model.intercept_
plt.plot(X, result)
plt.show()
