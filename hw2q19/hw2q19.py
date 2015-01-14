import numpy as np

# load_file
# =========
# load a file and get [X Y]
# 
def load_file(file_name):
    lines = open(file_name, 'r').readlines()
    return np.array([np.fromstring(line, dtype=float, sep=' ') for line in lines])

# solve_equation
# ==============
# return:
#   beta = (lambda * I + K)^(-1) * y
# 
def solve_equation(K, lamb, y):
    return np.linalg.solve(K + np.eye(K.shape[0]) * lamb, y)

# get_kernel_mat
# ==============
# return:
#   K[i,j] = kernel(X[i], X[j])
# note that K is symmetric
# 
def get_kernel_mat(X, kernel):
    m = X.shape[0]
    mat = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1):
            mat[i, j] = kernel(X[i], X[j])
    for i in range(m):
        for j in range(i + 1, m):
            mat[i, j] = mat[j, i]
    return mat

# kernel_hw2q19
# =============
# the RBF kernel
# 
def kernel_hw2q19(x1, x2, gamma):
    x = x1 - x2
    return np.exp(-gamma * np.dot(x, x))

# predict
# =======
# predict a vector x based on training set X, weights beta, and kernel
# 
def predict(x, X, beta, kernel):
    dots = np.array([kernel(x, v) for v in X])
    s = np.sum(dots * beta)
    return 2 * int(s > 0) - 1

if __name__ == '__main__':
    
    train = load_file('hw2_lssvm_all.dat')

    m_train = 400
    X_train = train[:m_train, :-1]
    Y_train = train[:m_train, -1].astype(int)

    X_test = train[m_train:, :-1]
    Y_test = train[m_train:, -1].astype(int)

    print '----------------------------------------'
    print '       Homework 2 Question 19, 20       '
    print '----------------------------------------'

    for gamma in [32, 2, 0.125]:
        for lamb in [0.001, 1, 1000]:
            print 'gamma  =', gamma
            print 'lambda =', lamb

            kernel = lambda x1, x2: kernel_hw2q19(x1, x2, gamma)
            
            K = get_kernel_mat(X_train, kernel)

            beta = solve_equation(K, lamb, Y_train)

            prediction = []
            for x in X_train:
                prediction.append(predict(x, X_train, beta, kernel))
            prediction = np.array(prediction)
            print 'E_in:'
            print float(np.sum(prediction != Y_train)) / len(Y_train)
            
            prediction = []
            for x in X_test:
                prediction.append(predict(x, X_train, beta, kernel))
            prediction = np.array(prediction)
            print 'E_out:'
            print float(np.sum(prediction != Y_test)) / len(Y_test)

            print

    plt.show()

