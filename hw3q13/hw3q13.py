import numpy as np
import matplotlib.pyplot as plt
import random

# plot_sign
# =========
# X: m * 2 matrix
# Y: m     array of +1 / -1
def plot_sign(X, Y):
    plt.plot(X[Y > 0, 0], X[Y > 0, 1], 'ro')
    plt.plot(X[Y < 0, 0], X[Y < 0, 1], 'go')

# load_file
# =========
# load a file and get [X Y]
# 
def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    return np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

# impurity
# ========
# when examples are split into two parts,
# add up the weighted Gini indices as the impurity
# 
def impurity(Y, left_idxs, right_idxs):
    nleftpos = np.count_nonzero(Y[left_idxs] == 1)
    nleftneg = np.count_nonzero(Y[left_idxs] == -1)
    if nleftpos + nleftneg != 0:
        gini_left = 1.0 - float(nleftpos ** 2 + nleftneg ** 2) / ((nleftpos + nleftneg) ** 2)
    else:
        gini_left = 0.0
    assert(gini_left <= 0.5)
    assert(gini_left >= 0.0)

    nrightpos = np.count_nonzero(Y[right_idxs] == 1)
    nrightneg = np.count_nonzero(Y[right_idxs] == -1)
    if nrightpos + nrightneg != 0:
        gini_right = 1.0 - float(nrightpos ** 2 + nrightneg ** 2) / ((nrightpos + nrightneg) ** 2)
    else:
        gini_right = 0.0
    assert(gini_right <= 0.5)
    assert(gini_right >= 0.0)

    return (nleftpos + nleftneg) * gini_left + (nrightpos + nrightneg) * gini_right

# DTNode
# ======
# x_i > thres => right
# x_i < thres => left
#
class DTNode:
    def __init__(self, i, thres, left_node, right_node):
        self.i = i
        self.thres = thres
        self.left_node = left_node
        self.right_node = right_node

    def predict(self, x):
        if x[self.i] < self.thres:
            return self.left_node.predict(x)
        else:
            return self.right_node.predict(x)

    def dump(self, level=0):
        print('  ' * level + 'i = %d, thres = %f' % (self.i, self.thres))
        self.left_node.dump(level + 1)
        self.right_node.dump(level + 1)

# DTLeaf
# ======
# return const
# 
class DTLeaf:
    def __init__(self, ans):
        self.ans = ans

    def predict(self, x):
        return self.ans

    def dump(self, level=0):
        print('  ' * level + '%d' % self.ans)

# RandomForest
# ============
# use a list of decision trees to vote for an answer
# 
class RandomForest:
    def __init__(self, dtrees):
        self.dtrees = dtrees

    def predict(self, x):
        predictions = [dtree.predict(x) for dtree in self.dtrees]
        if sum(predictions) > 0:
            return +1
        else:
            return -1

# get_split
# =========
# split the training examples into 2 parts to achieve the best purity
# 
def get_split(X_train, Y_train):
    m = len(Y_train)
    d = X_train.shape[1]

    best_i = 0
    best_thres = 0
    best_left_idxs = []
    best_right_idxs = []
    best_imp = m

    for i in range(d):
        features = X_train[:, i]
        sorted_feature_idxs = features.argsort()
        for split_idx in range(1, m):
            left_idxs = sorted_feature_idxs[:split_idx]
            right_idxs = sorted_feature_idxs[split_idx:]
            thres = 0.5 * (features[sorted_feature_idxs[split_idx]] + features[sorted_feature_idxs[split_idx - 1]])
            imp = impurity(Y_train, left_idxs, right_idxs)
            
            if imp < best_imp:
                best_imp = imp
                best_i = i
                best_thres = thres
                best_left_idxs = left_idxs
                best_right_idxs = right_idxs

    return best_i, best_thres, best_left_idxs, best_right_idxs

# train_pruned_dtree
# ==================
# get the best decision stump
# 
def train_pruned_dtree(X_train, Y_train):
    best_i, best_thres, best_left_idxs, best_right_idxs = get_split(X_train, Y_train)
    Y_train_left = Y_train[best_left_idxs]
    Y_train_right = Y_train[best_right_idxs]

    if np.sum(Y_train_left) > 0:
        left_node = DTLeaf(+1)
    else:
        left_node = DTLeaf(-1)

    if np.sum(Y_train_right) > 0:
        right_node = DTLeaf(+1)
    else:
        right_node = DTLeaf(-1)

    return DTNode(best_i, best_thres, left_node, right_node)

# train_dtree
# ===========
# train a C&RT
# X_train, Y_train shouldn't be empty
# 
def train_dtree(X_train, Y_train):

    # if training set is already pure, return constant
    if np.sum(Y_train) == -len(Y_train) or np.sum(Y_train) == len(Y_train):
        return DTLeaf(Y_train[0])

    i, thres, left_idxs, right_idxs = get_split(X_train, Y_train)

    X_train_left = X_train[left_idxs]
    Y_train_left = Y_train[left_idxs]
    assert(len(X_train_left) == len(Y_train_left))
    assert(len(X_train_left) > 0)
    dtree_left = train_dtree(X_train_left, Y_train_left)

    X_train_right = X_train[right_idxs]
    Y_train_right = Y_train[right_idxs]
    assert(len(X_train_right) == len(Y_train_right))
    assert(len(X_train_right) > 0)
    dtree_right = train_dtree(X_train_right, Y_train_right)

    return DTNode(i, thres, dtree_left, dtree_right)

# get_error
# =========
# 0/1 error
# 
def get_error(dtree, X, Y):
    Y_predict = map(dtree.predict, X)
    return float(np.count_nonzero(Y != Y_predict)) / len(Y)

# plot_train
# ==========
# plot the training examples and decision boundary
# 
def plot_train(dtree, X_train, Y_train):
    beg = 0
    end = 1
    npoints = 150
    xrange = np.linspace(beg, end, npoints)
    yrange = np.linspace(beg, end, npoints)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    zgrid = np.empty((npoints, npoints))
    for ix in range(npoints):
        for iy in range(npoints):
            zgrid[ix, iy] = dtree.predict([xgrid[ix, iy], ygrid[ix, iy]])

    plt.pcolor(xgrid, ygrid, zgrid)
    plot_sign(X_train, Y_train)

# plot_test
# =========
# plot the test examples and decision boundary
# 
def plot_test(dtree, X_test, Y_test):
    beg = 0
    end = 1
    npoints = 150
    xrange = np.linspace(beg, end, npoints)
    yrange = np.linspace(beg, end, npoints)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    zgrid = np.empty((npoints, npoints))
    for ix in range(npoints):
        for iy in range(npoints):
            zgrid[ix, iy] = dtree.predict([xgrid[ix, iy], ygrid[ix, iy]])

    plt.pcolor(xgrid, ygrid, zgrid)
    plot_sign(X_test, Y_test)

def hw3q13_14_15():
    train = load_file('hw3_train.dat')
    m = len(train)          # number of training examples
    d = train.shape[1] - 1  # number of feature dimensions
    X_train = train[:, :-1]
    Y_train = train[:, -1].astype(int)

    dtree = train_dtree(X_train, Y_train)
    print '----------------------------------------'
    print '         Homework 3 Question 13         '
    print '----------------------------------------'
    dtree.dump()
    print

    print '----------------------------------------'
    print '         Homework 3 Question 14         '
    print '----------------------------------------'
    print 'E_in =', get_error(dtree, X_train, Y_train)
    print

    test = load_file('hw3_test.dat')
    X_test = test[:, :-1]
    Y_test = test[:, -1].astype(int)
    print '----------------------------------------'
    print '         Homework 3 Question 15         '
    print '----------------------------------------'
    print 'E_out =', get_error(dtree, X_test, Y_test)
    print

    plt.figure()
    plt.suptitle('Decision Tree')
    plt.subplot(2, 1, 1)
    plot_train(dtree, X_train, Y_train)

    plt.subplot(2, 1, 2)
    plot_test(dtree, X_test, Y_test)
    plt.show()

def hw3q16():
    train = load_file('hw3_train.dat')
    m = len(train)          # number of training examples
    d = train.shape[1] - 1  # number of feature dimensions
    X_train = train[:, :-1]
    Y_train = train[:, -1].astype(int)

    dtrees = []
    errors = []
    for niter in range(1000): # should be 30000, but I found that 1000 is already okay.
        picked = []
        for idx in range(m):
            picked.append(random.randint(1, m - 1))
        cur_dtree = train_dtree(X_train[picked], Y_train[picked])
        dtrees.append(cur_dtree)
        errors.append(get_error(cur_dtree, X_train, Y_train))
    print '----------------------------------------'
    print '         Homework 3 Question 16         '
    print '----------------------------------------'
    print('avg(E_in) = %f' % np.average(errors))
    print

def hw3q17_18():
    train = load_file('hw3_train.dat')
    m = len(train)          # number of training examples
    d = train.shape[1] - 1  # number of feature dimensions
    X_train = train[:, :-1]
    Y_train = train[:, -1].astype(int)
    
    test = load_file('hw3_test.dat')
    X_test = test[:, :-1]
    Y_test = test[:, -1].astype(int)

    forests = []
    train_errors = []
    test_errors = []
    for niter in range(10):
        dtrees = []
        for niter in range(300):
            picked = []
            for idx in range(m):
                picked.append(random.randint(0, m - 1))
            cur_dtree = train_dtree(X_train[picked], Y_train[picked])
            dtrees.append(cur_dtree)
        cur_forest = RandomForest(dtrees)
        forests.append(cur_forest)
        train_errors.append(get_error(cur_forest, X_train, Y_train))
        test_errors.append(get_error(cur_forest, X_test, Y_test))
    print '----------------------------------------'
    print '         Homework 3 Question 17         '
    print '----------------------------------------'
    print('avg(E_in) = %f' % np.average(train_errors))
    print

    print '----------------------------------------'
    print '         Homework 3 Question 18         '
    print '----------------------------------------'
    print('avg(E_out) = %f' % np.average(test_errors))
    print

    plt.figure()
    plt.suptitle("The first random forest of C&RT's")
    plot_train(forests[0], X_train, Y_train)
    plt.show()

def hw3q19_20():
    train = load_file('hw3_train.dat')
    m = len(train)          # number of training examples
    d = train.shape[1] - 1  # number of feature dimensions
    X_train = train[:, :-1]
    Y_train = train[:, -1].astype(int)
    
    test = load_file('hw3_test.dat')
    X_test = test[:, :-1]
    Y_test = test[:, -1].astype(int)

    forests = []
    train_errors = []
    test_errors = []
    for niter in range(100):
        dtrees = []
        for niter in range(300):
            picked = []
            for idx in range(m):
                picked.append(random.randint(0, m - 1))
            cur_dtree = train_pruned_dtree(X_train[picked], Y_train[picked])
            dtrees.append(cur_dtree)
        cur_forest = RandomForest(dtrees)
        forests.append(cur_forest)
        train_errors.append(get_error(cur_forest, X_train, Y_train))
        test_errors.append(get_error(cur_forest, X_test, Y_test))

    print '----------------------------------------'
    print '         Homework 3 Question 19         '
    print '----------------------------------------'
    print('avg(E_in) = %f' % np.average(train_errors))
    print

    print '----------------------------------------'
    print '         Homework 3 Question 20         '
    print '----------------------------------------'
    print('avg(E_out) = %f' % np.average(test_errors))
    print

    plt.figure()
    plt.suptitle('The first random forest with pruned trees')
    plot_train(forests[0], X_train, Y_train)
    plt.show()

if __name__ == '__main__':
    hw3q13_14_15()
    hw3q16()
    hw3q17_18()
    hw3q19_20()

