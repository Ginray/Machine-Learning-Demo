# -*-coding:utf-8  -*-
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

learn_rate = 0.01
train_step = 10000
displat_step = 100

train_X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                    7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                    2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

n_sample = train_X.shape[0]

X = tf.placeholder(np.float32)
Y = tf.placeholder(np.float32)

W = tf.Variable(np.random.randn())
B = tf.Variable(np.random.randn())

pred = tf.add(tf.multiply(X, W), B)

cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_sample)
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    c = 'undifine'
    for i in range(train_step):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        if (i + 1) % displat_step == 0:
            c = sess.run(cost, feed_dict={X: x, Y: y})
            print 'Epoch={0},cost={1},w={2},b={3}'.format(i + 1, c, sess.run(W), sess.run(B))
            # 显示变换轨迹
            plt.plot(train_X, train_X * sess.run(W) + sess.run(B), 'r--')

    print 'Finish!'
    print 'Epoch={0},cost={1},w={2},b={3}'.format(i + 1, c, sess.run(W), sess.run(B))

    plt.scatter(train_X, train_Y)
    plt.plot(train_X, train_X * sess.run(W) + sess.run(B))
    plt.show()
