#coding:utf-8
import numpy as np
import pylab


def cal_cost(b,m,data):
    x=data[:,0]
    y=data[:,1]
    error = (y-m*x-b)**2
    total_error = np.sum(error,axis=0)
    #设axis=i，则numpy沿着第i个下标变化的方向进行操作。
    #简单的来记就是axis=0代表往跨行（down)，而axis=1代表跨列（across)
    return total_error/float(len(data))

def optimizer (init_b,init_m,num,data,learning_rate):
    b= init_b
    m= init_m

    for i in range(num):
        b,m= compute_gradient (b,m,data,learning_rate)
        if i % 100 == 0:
            print 'iter {0}:error={1}'.format(i, cal_cost(b, m, data))
    return [b, m]

def compute_gradient (b,m,data,learning_rate):

    N = float (len(data))

    x= data[:,0]
    y= data[:,1]
    b_gradient = -(2/N)*(y-m*x-b)
    b_gradient = np.sum(b_gradient,axis=0)
    m_gradient = -(2/N)*x*(y-m*x-b)
    m_gradient = np.sum(m_gradient,axis=0)

    b=b-(learning_rate*b_gradient)
    m=m-(learning_rate*m_gradient)
    return (b,m)


def plot_data(data,b,m):

    #plottting
    x = data[:,0]
    y = data[:,1]
    y_predict = m*x+b
    pylab.plot(x,y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()



def linear_regression():
    data= np.loadtxt('ex1data1.txt',delimiter=',')
    learning_rate = 0.001
    init_b= 0.0
    init_m= 0.0
    num = 1000

    print 'initial variables:\n initial_b = {0}\n intial_m = {1}\n error of begin = {2} \n' \
        .format(init_b, init_m, cal_cost(init_b, num, data))

    # optimizing b and m
    [b, m] = optimizer(init_b, init_m, num,data,learning_rate)

    # print final b m error
    print 'final formula parmaters:\n b = {1}\n m={2}\n error of end = {3} \n'.format(num, b, m,cal_cost(b, m, data))

    # plot result
    plot_data(data, b, m)


if __name__ == '__main__':
    linear_regression()