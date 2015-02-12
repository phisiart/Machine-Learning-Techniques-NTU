import numpy as np

def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    return np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

def knn(X_test, X_train, Y_train, k):
    ntrains, _ = X_train.shape
    ntests, ndims = X_test.shape
    
    # X_train : (ntrains, ndims) -> (1, ntrains, ndims)
    X_train = np.expand_dims(X_train, 0)

    # X_test : (ntests, ndims) -> (ntests, 1, ndims)
    X_test = np.expand_dims(X_test, 1)

    # length_squared : (ntests, ntrains)
    length_squared = np.sum((X_test - X_train) ** 2, axis=2)

    # sortedidx : (ntests, k)
    sortedidx = np.argsort(length_squared, axis=1)[:, :k]

    votes = np.mean(Y_train[sortedidx], axis=1)

    return 2 * (votes > 0) - 1

def hw4q15():
    print '----------------------------------------'
    print '         Homework 4 Question 15         '
    print '----------------------------------------'
    print 'avg(E_in) = %f' % np.mean(knn(X_train, X_train, Y_train, k=1) != Y_train)
    print
    
    print '----------------------------------------'
    print '         Homework 4 Question 16         '
    print '----------------------------------------'
    print 'avg(E_out) = %f' % np.mean(knn(X_test, X_train, Y_train, k=1) != Y_test)
    print

def hw4q17():
    print '----------------------------------------'
    print '         Homework 4 Question 17         '
    print '----------------------------------------'
    print 'avg(E_in) = %f' % np.mean(knn(X_train, X_train, Y_train, k=5) != Y_train)
    print
    
    print '----------------------------------------'
    print '         Homework 4 Question 18         '
    print '----------------------------------------'
    print 'avg(E_out) = %f' % np.mean(knn(X_test, X_train, Y_train, k=5) != Y_test)
    print

def main():
    global X_train
    global Y_train
    global X_test
    global Y_test

    train = load_file('hw4_knn_train.dat')
    X_train = train[:, :-1]
    Y_train = train[:, -1].astype(int)
    
    test = load_file('hw4_knn_test.dat')
    X_test = test[:, :-1]
    Y_test = test[:, -1].astype(int)

    hw4q15()
    hw4q17()

if __name__ == '__main__':
    main()