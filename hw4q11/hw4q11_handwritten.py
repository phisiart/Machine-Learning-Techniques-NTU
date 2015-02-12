import numpy as np
import matplotlib.pyplot as plt

# load_file
# =========
# load a file and get [X Y]
# 
def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    return np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

# plot_sign
# =========
# X: m * 2 matrix
# Y: m     array of +1 / -1
def plot_sign(X, Y):
    plt.plot(X[Y > 0, 0], X[Y > 0, 1], 'ro')
    plt.plot(X[Y < 0, 0], X[Y < 0, 1], 'go')

# plot_test
# =========
# plot the test examples and decision boundary
# 
def plot_test(func, X_test, Y_test):
    beg = -1.5
    end = 1.5
    npoints = 150
    xrange = np.linspace(beg, end, npoints)
    yrange = np.linspace(beg, end, npoints)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    zgrid = np.empty((npoints, npoints))
    for ix in range(npoints):
        for iy in range(npoints):
            zgrid[ix, iy] = (func([xgrid[ix, iy], ygrid[ix, iy]]) > 0).astype(int)

    plt.pcolor(xgrid, ygrid, zgrid)
    plot_sign(X_test, Y_test)

# d_tanh
# ======
# the derivative of tanh(x)
# 
def d_tanh(x):
    return 1.0 - np.tanh(x) ** 2

# activate
# ========
# activation function : x = tanh(s)
# s and x don't include the 1 bias term
# 
def activate(s):
    return np.tanh(s)

# W * x
# =====
# W[i, j] : i = 0, 1, ..., d(l-1)
#           j = 0, 1, ..., d(l) - 1
# W includes the bias
# 
# x[i]: i = 0, ... d(l-1) - 1
# x doesn't include the additional 1 term
# 
def layer(W, x):
    # return np.matrix(x) * np.matrix(W).T
    return np.array(np.matrix(np.append([1.0], x)) * np.matrix(W))

# forward
# =======
# Ws[l][i, j] : l = 0, ..., L-1
#               i = 0, ..., d(l-1) - 1
#               j = 0, ..., d(l) - 1
# ss[l][j] : l = 0, ..., L-1
#            j = 0, ..., d(l+1) - 1
# xs[l][i] : l = 0, ..., L
#            i = 0, ..., d(l)
#
def forward(Ws, x0):
    xs = [x0]
    x = x0
    ss = []
    for W in Ws:
        s = layer(W, x)
        ss.append(s)
        x = activate(s)
        xs.append(x)
    # print ss
    # print xs
    return ss, xs

# square_error
# ============
# debug function
# calculate the square error of one example
# (y - h(x))^2
# 
def square_error(Ws, x, y):
    _, xs = forward(Ws, x)
    return np.sum((xs[-1] - y) ** 2)

def decision_func(Ws):
    return lambda x : 2 * (forward(Ws, x)[1][-1] > 0) - 1

# error_01
# ========
# the 0/1 error of one test example
# 
def error_01(Ws, x, y):
    _, xs = forward(Ws, x)
    y_predict = 2 * (xs[-1] > 0) - 1
    assert y_predict == decision_func(Ws)(x)
    return int(y_predict != y)

# errors_01
# =========
# E_out
# 
def errors_01(Ws, X, Y):
    nsamples = len(X)
    assert(nsamples == len(Y))
    errors = []
    for isample in range(nsamples):
        errors.append(error_01(Ws, X[isample], Y[isample]))
    return np.mean(errors)

# square_errors
# =============
# the average square error of the test set
# 
def square_errors(Ws, X, Y):
    nsamples = len(X)
    assert(nsamples == len(Y))
    errors = []
    for isample in range(nsamples):
        errors.append(square_error(Ws, X[isample], Y[isample]))
    return np.mean(errors)

# backward_delta
# ==============
# get delta[l] from delta[l + 1]
# 
def backward_delta(W, delta, d_tanh_s):
    ret = np.array(np.matrix(delta) * np.matrix(W[1:, :]).T) * d_tanh_s
    assert(ret.shape == d_tanh_s.shape)
    return ret

# backward_derivatives
# ====================
# construct the gradients
# 
def backward_derivatives(Ws, deltas, xs):
    nlevels = len(Ws)
    return [np.matrix(np.append([1.0], xs[l])).T * np.matrix(deltas[l]) for l in range(nlevels)]

# backward_deltas
# ===============
# Ws[l][i, j] : l = 0, ..., L-1
#               i = 0, ..., d(l-1) - 1
#               j = 0, ..., d(l) - 1
# ss[l][j] : l = 0, ..., L-1
#            j = 0, ..., d(l)
#
# delta[l][j] = sum{k} delta[l+1][k] * W[l][j, k] * d_tanh(s[l][j])
def backward_deltas(Ws, ss, xs, y):
    nlevels = len(Ws)
    # print nlevels
    last_deltas = 2 * (xs[-1] - y) * d_tanh(ss[-1])
    deltas = [last_deltas]
    for l in reversed(range(nlevels - 1)):
        delta = backward_delta(Ws[l + 1], deltas[0], d_tanh(ss[l]))
        deltas = [delta] + deltas
    return deltas

# train_update
# ============
# update the weights based on the values calculated in backprop
# 
def train_update(Ws, derivates, learning_rate):
    nlevels = len(Ws)
    Ws_new = []
    assert(nlevels == len(derivates))
    for l in range(nlevels):
        Ws_new.append(Ws[l] - learning_rate * derivates[l])
    return Ws_new

# gradient_check
# ==============
# debug function
# 
def gradient_check(Ws, derivates, x, y):
    Ws_new = train_update(Ws, derivates, 0.00001)
    old_error = square_error(Ws, x, y)
    new_error = square_error(Ws_new, x, y)
    diff = sum([np.sum(derivate * 0.00001) for derivate in derivates])
    print 'old = %10.8f, new = %10.8f, diff = %10.8f, diff_true = %10.8f' % (old_error, new_error, diff, new_error - old_error)

# init_Ws
# =======
# randomly initialize weights
# 
def init_Ws(specs, r):
    Ws = []
    nlevels = len(specs) - 1
    for ilevel in range(nlevels):
        Ws.append(np.random.uniform(low=-r, high=r, size=(specs[ilevel] + 1, specs[ilevel + 1])))
    return Ws

# gradient_descent_step
# =====================
# one step of gradient descent
# 
def gradient_descent_step(Ws, x, y):
    ss, xs = forward(Ws, x)
    deltas = backward_deltas(Ws, ss, xs, y)
    derivates = backward_derivatives(Ws, deltas, xs)
    Ws = train_update(Ws, derivates, learning_rate)
    return Ws

def experiment(M, r, learning_rate, nexperiments):
    print '----------------------------------------'
    print 'M =', M, 'r =', r, 'rate =', learning_rate

    E_outs = []
    for _ in range(nexperiments):
        Ws = init_Ws([2, M, 1], r)

        for _ in range(50000):
            isample = np.random.choice(range(nsamples))
            x = X_train[isample]
            y = Y_train[isample]
            ss, xs = forward(Ws, x)
            deltas = backward_deltas(Ws, ss, xs, y)
            derivates = backward_derivatives(Ws, deltas, xs)
            Ws = train_update(Ws, derivates, learning_rate)

        E_out = errors_01(Ws, X_test, Y_test)
        print E_out
        E_outs.append(E_out)

    print 'avg(E_out) =', np.mean(E_outs)
    print '----------------------------------------'
    print

def experiment2():
    r = 0.1
    learning_rate = 0.01
    nexperiments = 100
    print '----------------------------------------'
    print '2 - 8 - 3 - 1', 'r =', r, 'rate =', learning_rate

    E_outs = []
    for _ in range(nexperiments):
        Ws = init_Ws([2, 8, 3, 1], r)

        for _ in range(50000):
            isample = np.random.choice(range(nsamples))
            x = X_train[isample]
            y = Y_train[isample]
            ss, xs = forward(Ws, x)
            deltas = backward_deltas(Ws, ss, xs, y)
            derivates = backward_derivatives(Ws, deltas, xs)
            Ws = train_update(Ws, derivates, learning_rate)

        E_out = errors_01(Ws, X_test, Y_test)
        print E_out
        E_outs.append(E_out)

    print 'avg(E_out) =', np.mean(E_outs)
    print '----------------------------------------'
    print

def main():
    train = load_file('hw4_nnet_train.dat')
    test = load_file('hw4_nnet_test.dat')

    global nsamples, X_train, Y_train
    nsamples = len(train)
    X_train = train[:, :-1]
    Y_train = train[:, -1]

    global X_test, Y_test
    X_test = test[:, :-1]
    Y_test = test[:, -1]

    nexperiments = 100 # instead of 500

    print '----------------------------------------'
    print '         Homework 4 Question 11         '
    print '----------------------------------------'
    r = 0.1
    learning_rate = 0.1
    experiment(1, r, learning_rate, nexperiments)
    experiment(6, r, learning_rate, nexperiments)
    experiment(11, r, learning_rate, nexperiments)
    experiment(16, r, learning_rate, nexperiments)
    experiment(21, r, learning_rate, nexperiments)

    print '----------------------------------------'
    print '         Homework 4 Question 12         '
    print '----------------------------------------'
    M = 3
    learning_rate = 0.1
    experiment(M, 0, learning_rate, nexperiments)
    experiment(M, 0.001, learning_rate, nexperiments)
    experiment(M, 0.1, learning_rate, nexperiments)
    experiment(M, 10, learning_rate, nexperiments)
    experiment(M, 1000, learning_rate, nexperiments)


    print '----------------------------------------'
    print '         Homework 4 Question 13         '
    print '----------------------------------------'
    r = 0.1
    M = 3
    experiment(M, r, 0.001, nexperiments)
    experiment(M, r, 0.01, nexperiments)
    experiment(M, r, 0.1, nexperiments)
    experiment(M, r, 1, nexperiments)
    experiment(M, r, 10, nexperiments)

    print '----------------------------------------'
    print '         Homework 4 Question 14         '
    print '----------------------------------------'
    experiment2()


if __name__ == '__main__':
    main()
