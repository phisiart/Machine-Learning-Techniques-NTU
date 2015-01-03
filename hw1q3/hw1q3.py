import sklearn
import numpy as np
import cvxopt as opt
import matplotlib.pyplot as plt

# Quadratic Programming in cvxopt
# ===============================
# min 1/2 * x' * P * x + q' * x
# subject to G * x <= h
#            A * x = b


# svm_hard_linear
# ===============
# use the QP solver to solve a hard-margin SVM primal problem
# input: X, Y
#   elements of Y must either be 1 or -1
#
# example:
#   X = np.array([[0, 0], [1, 0], [0, 2], [-2, 0]])
#   Y = np.array([1, 1, -1, -1])
#
# return b, w
#
# prediction(x) = sign(w'x + b)
#
def svm_hard_linear(X, Y):
    m, d = X.shape

    A = np.array(np.bmat([np.ones(m)[:, np.newaxis], X]))
    A = A * Y[:,np.newaxis]

    P = opt.matrix(np.diag([0] + [1] * d), tc='d')
    q = opt.matrix([0] * (d + 1), tc='d')
    G = opt.matrix(-A, tc='d')
    h = opt.matrix(-np.ones(m), tc='d')

    sol = opt.solvers.qp(P, q, G, h)
    b, w = sol['x'][0], (sol['x'][1:]).T
    return (b, w)


# svm_hard_kernel
# ===============
# use the QP solver to solve a hard-margin kernel SVM dual problem
# input: X, Y, kernel
#   elements of Y must either be 1 or -1
#   kernel is a function that takes two numpy arrays and return a number
#
# example:
#   X = np.array([[1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2,0]])
#   Y = np.array([   -1,    -1,     -1,      1,     1,      1,      1])
#   kernel = np.dot
#
# return:
#   { 'alpha': a, 'b': b }
#
# prediction(x) = sign(b + sum[i = 1..m](a[i] * Y[i] * kernel(X[i], x)))
#
def svm_hard_kernel(X, Y, kernel):
    m, d = X.shape
    K = np.ones((m, m))

    # construct the kernel matrix: K[i, j] = Y[i] * Y[j] * K(X[i], X[j])
    # note that K is symmetric
    for i in range(m):
        for j in range(i, m):
            K[i, j] = Y[i] * Y[j] * kernel(X[i], X[j])
    for i in range(m):
        for j in range(i):
            K[i, j] = K[j,i]

    # 1/2 x'Px + q'x <-> 1/2 x'Kx - 1'x
    P = opt.matrix(K, tc='d')
    q = opt.matrix([-1] * m, tc='d')

    # Ax = b <-> y' * a = 0
    A = opt.matrix(Y[:,np.newaxis].T, tc='d')
    # A = opt.matrix(Y, tc='d')
    b = opt.matrix([0], tc='d')

    # Gx <= h  <-> -a <= 0
    G = opt.matrix(-np.identity(m), tc='d')
    h = opt.matrix([0] * m, tc='d')

    # sol = {'x': 1}
    sol = opt.solvers.qp(P, q, G, h, A, b)

    alpha = sol['x']

    # s = index of a free support vector
    s = np.argmax(alpha)
    b = Y[s] - np.sum([(alpha[n] * Y[n]) * kernel(X[n], X[s]) for n in range(m)])

    return { 'alpha': alpha.T, 'b': b }

# svm_test
# ========
# test a simple SVM problem in the course video
def svm_test():
    print '----------svm_test----------'
    X = np.array([[0, 0], [1, 0], [0, 2], [-2, 0]])
    Y = np.array([1, 1, -1, -1])
    b, w = svm_hard_linear(X, Y)
    print 'Primal problem returns:'
    print 'b =', b
    print 'w =', w

    ret = svm_hard_kernel(X, Y, np.dot)
    print 'Dual problem returns:'
    print 'b = ', ret['b']
    print 'alpha = ', ret['alpha']


# hw1q3_kernel
# ============
# this is the kernel that we will use in homework 1 question 3
# K(x, y) = (1 + x'y)^2
def hw1q3_kernel(x1, x2):
    x = 1 + np.dot(x1, x2)
    return x ** 2


# hw1q3
# =====
# the code for homework 1 question 3
#
def hw1q3():
    X = np.array([
        [1, 0],
        [0, 1],
        [0, -1],
        [-1, 0],
        [0, 2],
        [0, -2],
        [-2, 0]
    ])

    Y = np.array([-1, -1, -1, 1, 1, 1, 1])

    ret = svm_hard_kernel(X, Y, hw1q3_kernel)
    alpha = ret['alpha']
    b = ret['b']
    
    print 'HW1Q3 answers:'
    print 'b =', b
    print 'alpha =', alpha
    
    plt.suptitle('red circles are +1 examples, blue crosses are -1 examples')
    plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'ro')
    plt.plot(X[Y == -1, 0], X[Y == -1, 1], 'bx')
    
    beg = -3
    end = 3
    npoints = 100
    
    xrange = np.linspace(beg, end, npoints)
    yrange = np.linspace(beg, end, npoints)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    zgrid = np.empty((npoints, npoints))

    for i in range(npoints):
        for j in range(npoints):
            pos = np.array([xgrid[i,j], ygrid[i,j]])
            zgrid[i][j] = b + np.sum([(alpha[n] * Y[n]) * hw1q3_kernel(X[n], pos) for n in range(len(X))])

    plt.contour(xgrid, ygrid, zgrid, 1)
    plt.show()


if __name__ == '__main__':
    hw1q3()



