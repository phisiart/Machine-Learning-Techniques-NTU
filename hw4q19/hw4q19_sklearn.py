from sklearn.cluster import KMeans
import numpy as np

def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    return np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

def experiment(nexperiments, nclusters):
    E_ins = []
    for _ in range(nexperiments):
        kmeans = KMeans(nclusters, max_iter=300, n_init=1, init='random')
        kmeans.fit(X_train)
        score = kmeans.score(X_train)
        E_in = -score / nsamples
        E_ins.append(E_in)
    return E_ins

if __name__ == '__main__':
    X_train = load_file('hw4_kmeans_train.dat')
    nsamples, ndims = X_train.shape

    E_ins = experiment(nexperiments=500, nclusters=2)

    print '----------------------------------------'
    print '         Homework 4 Question 19         '
    print '----------------------------------------'
    print 'avg(E_in) = %f' % np.mean(E_ins)
    print

    E_ins = experiment(nexperiments=500, nclusters=10)

    print '----------------------------------------'
    print '         Homework 4 Question 20         '
    print '----------------------------------------'
    print 'avg(E_in) = %f' % np.mean(E_ins)
    print