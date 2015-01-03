import sklearn.svm
import numpy as np
import matplotlib.pyplot as plt

train = np.array([np.fromstring(line, dtype=float, sep=' ') for line in open('features.train', 'r').readlines()])
X_train = train[:, 1:]
Y_train = train[:, 0].astype(int)

test = np.array([np.fromstring(line, dtype=float, sep=' ') for line in open('features.test', 'r').readlines()])
X_test = test[:, 1:]
Y_test = test[:, 0].astype(int)

def plot_samples(X, Y):
    plt.plot(X[Y==False, 0], X[Y==False, 1], 'r.')
    plt.plot(X[Y, 0], X[Y, 1], 'g.')

def plot_01(X, Y):
    plt.plot(X[Y == 0, 0], X[Y == 0, 1], 'r.')
    plt.plot(X[Y == 1, 0], X[Y == 1, 1], 'g.')

def count_error(Y, Y_predict):
    return np.count_nonzero(Y != Y_predict)

# hw1q15
# ======
# use a linear kernel in a soft-margin SVM and find || w ||
def hw1q15():
    svm = sklearn.svm.SVC(C=0.01, kernel='linear', shrinking=False, verbose=True)

    X_train_0 = X_train
    Y_train_0 = (Y_train == 0).astype(int)

    svm.fit(X_train_0, Y_train_0)

    w = svm.coef_[0]
    b = svm.intercept_[0]
    
    print 'w =', w
    print 'norm(w) =', np.linalg.norm(w, ord=2)
    print 'b =', b


# hw1q16
# ======
def hw1q16():
    print '----------------------------------------'
    print '         Homework 1 Question 16         '
    print '----------------------------------------'

    # polynomial kernel: (coef0 + gamma * x1.T * x2) ** degree

    for idx in (0, 2, 4, 6, 8):
        svm = sklearn.svm.SVC(C=0.01, kernel='poly', degree=2, gamma=1, coef0=1, tol=1e-4, shrinking=True, verbose=False)

        Y_train_i = (Y_train == idx).astype(int)

        svm.fit(X_train, Y_train_i)
        Y_predict_i = svm.predict(X_train)

        support = svm.support_
        coef = svm.dual_coef_[0]
        b = svm.intercept_[0]
        E_in = np.count_nonzero(Y_train_i != Y_predict_i)

        print 'For class %d:' % (idx)        
        print 'sum(alpha) =', np.sum(np.abs(coef))
        print 'b =', b
        print 'E_in =', E_in

        fig = plt.figure()
        # plt.suptitle('%d vs rest' % (idx))
        plt.subplot(311)
        plt.title('Training data: green +, red -')
        plot_01(X_train, Y_train_i)
        plt.tick_params(axis='x', labelbottom='off')
        
        plt.subplot(312)
        plt.title('Prediction: green +, red -')
        plot_01(X_train, Y_predict_i)
        plt.tick_params(axis='x', labelbottom='off')

        plt.subplot(313)
        plt.title('Support vectors: blue')
        plt.plot(X_train[:, 0], X_train[:, 1], 'r.')
        plt.plot(X_train[support, 0], X_train[support, 1], 'b.')

    plt.show()


# hw1q18
# ======
def hw1q18():
    print '----------------------------------------'
    print '         Homework 1 Question 18         '
    print '----------------------------------------'

    Y_train_0 = (Y_train == 0).astype(int)
    Y_test_0 = (Y_test == 0).astype(int)

    print 'in the training set:'
    print 'n(+) =', np.count_nonzero(Y_train_0 == 1), 'n(-) =', np.count_nonzero(Y_train_0 == 0)

    print 'in the test set:'
    print 'n(+) =', np.count_nonzero(Y_test_0 == 1), 'n(-) =', np.count_nonzero(Y_test_0 == 0)


    for C in (0.001, 0.01, 0.1, 1, 10):
        svm = sklearn.svm.SVC(C=C, kernel='rbf', gamma=100, tol=1e-7, shrinking=True, verbose=False)
        svm.fit(X_train, Y_train_0)

        print '----------------------------------------'
        print 'C =', C

        support = svm.support_
        coef = svm.dual_coef_[0]
        b = svm.intercept_[0]

        print 'nSV =', len(support)
        Y_predict = svm.predict(X_test)

        print 'in the prediction:'
        print 'n(+) =', np.count_nonzero(Y_predict == 1), 'n(-) =', np.count_nonzero(Y_predict == 0)

        print 'E_out =', np.count_nonzero(Y_test_0 != Y_predict)
        print

        fig = plt.figure()
        plt.suptitle('C =' + str(C))
        plt.subplot(311)
        plt.title('Training data: green +, red -')
        plot_01(X_train, Y_train_0)
        plt.tick_params(axis='x', labelbottom='off')
        
        plt.subplot(312)
        plt.title('Prediction on test data: green +, red -')
        plot_01(X_test, Y_predict)
        plt.tick_params(axis='x', labelbottom='off')

        plt.subplot(313)
        plt.title('Support vectors: blue')
        plt.plot(X_train[:, 0], X_train[:, 1], 'r.')
        plt.plot(X_train[support, 0], X_train[support, 1], 'b.')

    plt.show()


# hw1q19
# ======
def hw1q19():
    print '----------------------------------------'
    print '         Homework 1 Question 19         '
    print '----------------------------------------'

    Y_train_0 = (Y_train == 0).astype(int)
    Y_test_0 = (Y_test == 0).astype(int)

    for gamma in (1, 10, 100, 1000, 10000):
        svm = sklearn.svm.SVC(C=0.1, kernel='rbf', gamma=gamma, tol=1e-7, shrinking=True, verbose=False)
        svm.fit(X_train, Y_train_0)
        print '----------------------------------------'
        print 'gamma =', gamma
        Y_predict_0 = svm.predict(X_test)
        print 'in the prediction:'
        print 'n(+) =', np.count_nonzero(Y_predict_0 == 1), 'n(-) =', np.count_nonzero(Y_predict_0 == 0)

        print 'E_out =', np.count_nonzero(Y_test_0 != Y_predict_0)
        print


# hw1q20
# ======
def hw1q20():
    print '----------------------------------------'
    print '         Homework 1 Question 20         '
    print '----------------------------------------'

    Y_train_0 = (Y_train == 0).astype(int)

    C = 0.1
    m = len(Y_train_0)
    gammas = [1, 10, 100, 1000, 10000]
    counts = [0] * len(gammas)

    for nrun in range(10):
        print 'run', nrun

        # generate a random order of m indices
        arr = np.arange(m)
        np.random.shuffle(arr)

        # pick 1000 for cross validation
        X_curval_0 = X_train[arr[:1000]]
        Y_curval_0 = Y_train_0[arr[:1000]]
        X_curtrain_0 = X_train[arr[1000:]]
        Y_curtrain_0 = Y_train_0[arr[1000:]]

        E_vals = [0.0] * len(gammas)
        for i in range(len(gammas)):
            gamma = gammas[i]

            svm = sklearn.svm.SVC(C=C, kernel='rbf', gamma=gamma, tol=1e-3, shrinking=True, verbose=False)
            svm.fit(X_curtrain_0, Y_curtrain_0)
            Y_curpredict_0 = svm.predict(X_curval_0)
            E_val = np.count_nonzero(Y_curval_0 != Y_curpredict_0)

            E_vals[i] = E_val

        counts[np.argmin(E_vals)] += 1

    for i in range(len(gammas)):
        print 'gamma', gammas[i], 'got picked', counts[i], 'times'


# svm_test
# ========
# a simple test on sklearn's SVM
def svm_test():
    X_train = np.array([[0, 0], [1, 0], [0, 2], [-2, 0]])
    Y_train = np.array([1, 1, 0, 0])
    svm = sklearn.svm.SVC(C=100000, kernel='linear', shrinking=False, verbose=False)
    svm.fit(X_train, Y_train)
    Y_predict = svm.predict(X_train)
    print Y_predict
    b = svm.intercept_[0]
    print b

    plt.figure()
    plt.suptitle('svm test')
    plt.subplot(211)
    plot_01(X_train, Y_train)
    plt.subplot(212)
    plot_01(X_train, Y_predict)
    plt.plot(X_train[Y_predict == 0, 0], X_train[Y_predict == 0, 1], 'ro')
    plt.plot(X_train[Y_predict == 1, 0], X_train[Y_predict == 1, 1], 'go')
    plt.show()

hw1q15()
hw1q16()
hw1q18()
hw1q19()
hw1q20()
