# coding: utf-8
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ', data.shape)
    print(data[1:6, :])
    return (data)


def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:, 2] == 0  # 表示=0的样本的序号
    pos = data[:, 2] == 1

    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True);
    plt.show()


data = loaddata('data1.txt', ',')

'''
numpy.c_ = <numpy.lib.index_tricks.CClass object>
将切片对象沿第二个轴（按列）转换为连接。
'''
X = np.c_[np.ones((data.shape[0], 1)), data[:, 0:2]]
y = np.c_[data[:, 2]]

plotData(data, 'Exam 1 score', 'Exam 2 score', 'Pass', 'Fail')


# 定义sigmoid函数
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


# 定义损失函数
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    J = -1.0 * (1.0 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y))
    if np.isnan(J[0]):
        return (np.inf)
    return J[0]


# 求解梯度
def gradient(theta, X, y):
    m = y.size
    '''
    关于reshape:大意是说，数组新的shape属性应该要与原来的配套，如果等于-1的话，
    那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    也就是说，先前我们不知道z的shape属性是多少，但是想让z变成只有一列，行数不知道多少，
    通过`z.reshape(-1,1)`，Numpy自动计算出有12行，
    新的数组shape属性为(16, 1)，与原来的(4, 4)配套。
    '''
    h = sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1.0 / m) * X.T.dot(h - y)
    return (grad.flatten())


initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)
'''(useage for minimize)

fun : callable

    The objective function to be minimized. Must be in the form f(x, *args). The optimizing argument, x, is a 1-D array of points, and args is a tuple of any additional fixed parameters needed to completely specify the function.

x0 : ndarray

    Initial guess. len(x0) is the dimensionality of the minimization problem.

args : tuple, optional

    Extra arguments passed to the objective function and its derivatives (Jacobian, Hessian).

method : str or callable, optional

    Type of solver. Should be one of

            ‘Nelder-Mead’ (see here)
            ‘Powell’ (see here)
            ‘CG’ (see here)
            ‘BFGS’ (see here)
            ‘Newton-CG’ (see here)
            ‘L-BFGS-B’ (see here)
            ‘TNC’ (see here)
            ‘COBYLA’ (see here)
            ‘SLSQP’ (see here)
            ‘dogleg’ (see here)
            ‘trust-ncg’ (see here)
            ‘trust-exact’ (see here)
            ‘trust-krylov’ (see here)
            custom - a callable object (added in version 0.14.0), see below for description.

    If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP, depending if the problem has constraints or bounds.

jac : bool or callable, optional

    Jacobian (gradient) of objective function. Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-region-exact. If jac is a Boolean and is True, fun is assumed to return the gradient along with the objective function. If False, the gradient will be estimated numerically. jac can also be a callable returning the gradient of the objective. In this case, it must accept the same arguments as fun.

hess, hessp : callable, optional

    Hessian (matrix of second-order derivatives) of objective function or Hessian of objective function times an arbitrary vector p. Only for Newton-CG, dogleg, trust-ncg, trust-krylov, trust-region-exact. Only one of hessp or hess needs to be given. If hess is provided, then hessp will be ignored. If neither hess nor hessp is provided, then the Hessian product will be approximated using finite differences on jac. hessp must compute the Hessian times an arbitrary vector.

bounds : sequence, optional

    Bounds for variables (only for L-BFGS-B, TNC and SLSQP). (min, max) pairs for each element in x, defining the bounds on that parameter. Use None for one of min or max when there is no bound in that direction.

constraints : dict or sequence of dict, optional

    Constraints definition (only for COBYLA and SLSQP). Each constraint is defined in a dictionary with fields:

        type : str

            Constraint type: ‘eq’ for equality, ‘ineq’ for inequality.
        fun : callable

            The function defining the constraint.
        jac : callable, optional

            The Jacobian of fun (only for SLSQP).
        args : sequence, optional

            Extra arguments to be passed to the function and Jacobian.

    Equality constraint means that the constraint function result is to be zero whereas inequality means that it is to be non-negative. Note that COBYLA only supports inequality constraints.

tol : float, optional

    Tolerance for termination. For detailed control, use solver-specific options.

options : dict, optional

    A dictionary of solver options. All methods accept the following generic options:

        maxiter : int

            Maximum number of iterations to perform.
        disp : bool

            Set to True to print convergence messages.

    For method-specific options, see show_options.

callback : callable, optional

    Called after each iteration, as callback(xk), where xk is the current parameter vector.
'''
res = minimize(costFunction, initial_theta, args=(X, y), jac=gradient, options={'maxiter': 400})
print res


def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return (p.astype('int'))


sigmoid(np.array([1, 45, 85]).dot(res.x.T))

plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:, 1].min(), X[:, 1].max(),
x2_min, x2_max = X[:, 2].min(), X[:, 2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0], 1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');
plt.show()

