import numpy as np
import matplotlib.pyplot as plt

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


# predict
# =======
# s * (2 * (x[i] > thres) - 1.0)
# 
def predict(X, s, i, thres):
    return (s * (2 * (X[:, i] > thres) - 1)).astype(int)


# predict_final
# =============
# aggregate all g's
# 
def predict_final(X, gs):
    # g_t = (s, i, thres)
    predictions = np.array([predict(X, s, i, thres) * alpha for (s, i, thres, alpha) in gs])
    prediction = np.sum(predictions, 0)
    prediction = (2 * (prediction > 0) - 1).astype(int)
    return prediction


# update_u_alpha
# ==============
# return: (new u, alpha, epsilon)
# the epsilon is just for hw2q16
# 
# in AdaBoost, we need to
#   1) update the weight vector u[0..m] so that the errors are more heavily weighted
#   2) get a weight alpha[t] for the current g[t], so that the final prediction function makes use of all g[t]'s
# 
def update_u_alpha(Y_predict, Y, u):

    # correctness array, True for error
    errs = (Y_predict != Y)

    # incorrect rate
    epsilon = float(np.sum(errs * u)) / (np.sum(u))

    coeff_incorrect = np.sqrt((1 - epsilon) / epsilon)
    coeff_correct = np.sqrt(epsilon / (1 - epsilon))

    # the coef's applied to u
    coef = [coeff_incorrect if err else coeff_correct for err in errs]
    
    # (new u, alpha, epsilon)
    return (u * coef, np.log(coeff_incorrect), epsilon)
    

# hw2q12-q18
# ===========
if __name__ == '__main__':

    train = load_file('hw2_adaboost_train.dat')
    m = len(train)          # number of training examples
    d = train.shape[1] - 1  # feature dimension

    X_train = train[:, :-1]
    Y_train = train[:, -1].astype(int)

    test = load_file('hw2_adaboost_test.dat')
    m_test = len(test)
    X_test = test[:, :-1]
    Y_test = test[:, -1].astype(int)

    u = np.array([1.0 / m] * m) # initialize weight vector u with a uniform distribution
    gs = []                     # the different prediction functions
    epsilons = []
    niters = 300

    for t in range(1, niters + 1):
    
        if t == 1 or t == 2 or t == niters:
            print '----------------------------------------'
            print '       Homework 2 Question 14, 15       '
            print '----------------------------------------'
            print 'iteration', t
            print '  sum(u) =', np.sum(u)

        min_err = np.inf
        best_s = 0
        best_thres = -np.inf
        best_i = 0

        for i in range(d):

            # sort the rows by feature (column) i
            sortidx = train[:, i].argsort()

            train_sorted = train[sortidx]
            u_sorted = u[sortidx]
            sum_u_sorted = sum(u_sorted)

            # acc[n] = sum of [u_0 * y_0, u_n * y_n)
            # n = 0 .. m
            acc = [0]
            for n in range(m):
                acc.append(acc[-1] + train_sorted[n, -1] * u_sorted[n])

            # s = +1:
            # err = sum(0.5 * u[i] * (1 + y[i])) + sum(0.5 * u[j] * (1 - y[j]))
            #     = sum(0.5 * u[k]) + sum(0.5 * u[i] * y[i]) - sum(0.5 * u[j] * y[j])
            err_pos = np.array([0.5 * (acc[split] + sum_u_sorted - (acc[m] - acc[split])) for split in range(m)])

            # s = -1:
            # err = sum(0.5 * u[i] * (1 - y[i])) + sum(0.5 * u[j] * (1 + y[j]))
            #     = sum(0.5 * u[k]) - sum(0.5 * u[i] * y[i]) + sum(0.5 * u[j] * y[j])
            err_neg = np.array([0.5 * (-acc[split] + sum_u_sorted + (acc[m] - acc[split])) for split in range(m)])

            err_pos_argmin = np.argmin(err_pos)
            err_neg_argmin = np.argmin(err_neg)

            if err_pos[err_pos_argmin] < err_neg[err_neg_argmin]:
                s = 1.0
                if err_pos_argmin == 0:
                    thres = -np.inf
                else:
                    thres = 0.5 * (train_sorted[err_pos_argmin, i] + train_sorted[err_pos_argmin - 1, i])
                err = err_pos[err_pos_argmin]
            else:
                s = -1.0
                if err_neg_argmin == 0:
                    thres = -np.inf
                else:
                    thres = 0.5 * (train_sorted[err_neg_argmin, i] + train_sorted[err_neg_argmin - 1, i])
                err = err_neg[err_neg_argmin]

            if err < min_err:
                best_s = s
                best_thres = thres
                min_err = err
                best_i = i

        # for i in range(d) -- end

        # print 'best model in this iteration:'
        # print 's =', best_s
        # print 'i =', best_i
        # print 'thres =', best_thres
        # print 'err =', min_err

        # now we are updating u and acquiring alpha
        Y_predict = predict(X_train, best_s, best_i, best_thres)

        if t == 1:
            print '----------------------------------------'
            print '         Homework 1 Question 12         '
            print '----------------------------------------'
            print 'E_in(g1):'
            print float(np.sum(Y_predict != Y_train)) / m
            print

        u, alpha, epsilon = update_u_alpha(Y_predict, Y_train, u)

        # epsilons are just for question 16
        epsilons.append(epsilon)

        # add this model
        gs.append((best_s, best_i, best_thres, alpha))


    print 'training finished'

    print '----------------------------------------'
    print '         Homework 1 Question 13         '
    print '----------------------------------------'
    Y_predict = predict_final(X_train, gs)
    print 'E_in(G):'
    print float(np.sum(Y_predict != Y_train)) / m
    print

    print '----------------------------------------'
    print '         Homework 2 Question 16         '
    print '----------------------------------------'
    print 'min(epsilons) =', min(epsilons)

    # let's plot the decision function
    beg = 0
    end = 1
    npoints = 150
    xrange = np.linspace(beg, end, npoints)
    yrange = np.linspace(beg, end, npoints)
    xgrid, ygrid = np.meshgrid(xrange, yrange)
    zgrid = np.empty((npoints, npoints))

    X_grid = np.array([xgrid.reshape(npoints * npoints), ygrid.reshape(npoints * npoints)]).T
    zgrid = predict_final(X_grid, gs).reshape(npoints, npoints)

    plt.figure()
    plt.suptitle('AdaBoost')

    plt.subplot(211)
    plt.pcolor(xgrid, ygrid, zgrid)
    plot_sign(X_train, Y_train)
    
    plt.subplot(212)
    plt.pcolor(xgrid, ygrid, zgrid)
    plot_sign(X_test, Y_test)

    plt.show()

    print '----------------------------------------'
    print '         Homework 2 Question 17         '
    print '----------------------------------------'
    Y_test_predict = predict(X_test, *(gs[0][:3]))
    print 'E_out(g1):'
    print float(np.sum(Y_test_predict != Y_test)) / m_test

    print '----------------------------------------'
    print '         Homework 2 Question 18         '
    print '----------------------------------------'
    Y_test_predict = predict_final(X_test, gs)
    print 'E_out(G):'
    print float(np.sum(Y_test_predict != Y_test)) / m_test

