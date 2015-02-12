import numpy as np

def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    return np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

def init_centroids(X, nclusters):
    nsamples, ndims = X.shape
    return X[np.random.choice(nsamples, nclusters, replace=False)]

def update_assignments(X, centroids):
    nsamples, ndims = X.shape
    nclusters = len(centroids)

    # X : (nsamples, ndims) -> (1, nsamples, ndims)
    X = np.expand_dims(X, 0)

    # centroids : (nclusters, ndims) -> (nclusters, 1, ndims)
    centroids = np.expand_dims(centroids, 1)

    # diff : (nclusters, nsamples, ndims)
    diff = X - centroids

    # length_squared : (nclusters, nsamples)
    length_squared = np.sum(diff ** 2, 2)

    # assignments : (nsamples)
    assignments = np.argmin(length_squared, 0)

    return assignments

def update_centroids(X, assignments, nclusters):
    nsamples, ndims = X.shape

    centroids = []
    for icluster in range(nclusters):
        if len(X[assignments == icluster, :]) > 0:
            centroids.append(np.mean(X[assignments == icluster, :], 0))
        else:
            print assignments
            assert(0)

    return np.array(centroids)

def experiment(X, nexperiments, nclusters):
    nsamples, ndims = X.shape

    E_ins = []
    for i in range(nexperiments):
        niters = 300
        centroids = init_centroids(X, nclusters)
        assignments = update_assignments(X, centroids)
        for i in range(niters):
            centroids = update_centroids(X, assignments, nclusters)
            old_assignments = assignments
            assignments = update_assignments(X, centroids)
            if np.sum(old_assignments == assignments) == nsamples:
                break

        length_squared = np.sum((X - centroids[assignments, :]) ** 2, 1)
        E_in = np.mean(length_squared)
        E_ins.append(E_in)
    
    return np.array(E_ins)

def hw4q19():
    X_train = load_file('hw4_kmeans_train.dat')
    nsamples, ndims = X_train.shape

    E_ins = experiment(X_train, nexperiments=500, nclusters=2)

    print '----------------------------------------'
    print '         Homework 4 Question 19         '
    print '----------------------------------------'
    print 'avg(E_in) = %f' % np.mean(E_ins)
    print

def hw4q20():
    X_train = load_file('hw4_kmeans_train.dat')
    nsamples, ndims = X_train.shape

    E_ins = experiment(X_train, nexperiments=500, nclusters=10)

    print '----------------------------------------'
    print '         Homework 4 Question 19         '
    print '----------------------------------------'
    print 'avg(E_in) = %f' % np.mean(E_ins)
    print

if __name__ == '__main__':
    hw4q19()
    hw4q20()
