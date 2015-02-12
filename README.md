## Machine Learning Techniques - TaiwanU

This repository contains my code for the assignments in the '[Machine Learning Techniques](https://www.coursera.org/course/ntumltwo)' course from National Taiwan University on [Coursera](https://www.coursera.org/).


### LibSVM
In the `libsvm` folder, I put two files: `svm.h` and `svm.c`. The source of this library can be found [here](https://github.com/cjlin1/libsvm). I always put these two files along with my C++ code files and `#include "svm.h"` to use the library.

### Python libraries used
`cvxopt`: for Quadratic Programming

`scipy`: a stack of libraries containing `numpy`, `matplotlib`, `sklearn`, ...


## Homework 1
### Question 3, 4
`/hw1q3/hw1q3.py`

`sklearn`, `numpy`, `cvxopt`, `matplotlib` used

In this question, I implemented hard-margin linear SVM (primal) and kernel SVM (dual) using a QP solver.

The boundary is plotted.

### Question 15
`/hw1q15_cpp/`: A c++ version of question 15 in function `hw1q15()`, using LibSVM.

`/hw1q15_py/hw1q15.py`: A python version of question 15 in function `hw1q15`, using sklearn.

### Question 16
`/hw1q15_cpp/`: A c++ version of question 15 in function `hw1q16()`, using LibSVM.

`/hw1q15_py/hw1q15.py`: A python version of question 15 in function `hw1q16`, using sklearn.

Note: in `/hw1q15_cpp/`, use

    make
    ./hw1q15
    
will run both question 15 and 16.

### Question 18, 19, 20
`/hw1q15/hw1q15.py`: these three questions in functions `hw1q18`, `hw1q19`, `hw1q20`.

Note: in `/hw1q15_py/`, use

	python hw1q15.py

will run all questions 15, 16, 18, 19, 20.

## Homework 2
### Question 12 - 18
`/hw2q12/hw2q12.py`: an AdaBoost with stumps implemented

`numpy`, `matplotlib` used

Note: in `/hw2q12/`, simply use

    python hw2q12.py
    
will run the AdaBoost and print out the answers to all questions 12 - 18. The script will also draw the decision boundary.

### Question 19, 20
`/hw2q19/hw2q19.py`: a kernel ridge regression implemented

`numpy` used

Note: in `/hw2q19/`, use

    python hw2q19.py
    
will run the regression and print out the answers to questions 19, 20.

## Homework 3
`/hw3q13/hw3q13.py`: solutions to all questions in homework 3 in this file.

`numpy`, `matplotlib` used

Note: in `/hw3q13/`, use

    python hw3q13.py

will run all solutions to questions 13 - 20.

### Question 13 - 15
`hw3q13_14_15()`: trains a hand-written decision tree; dumps the branches; prints out E_in and E_out; plots the trained decision boundary.

### Question 16
`hw3q16()`: trains a lot of decision trees using bagging; computes the average E_in.

note that I only trained 1000 trees instead of 30000, since they're already enough to get a quite stable average E_in

### Question 17, 18
`hw3q17_18()`: trains a lot of random forests with C&RT's; computes the average E_in and E_out; plots the first trained random forest as an example.

### Question 19, 20
`hw3q19_20`: trains a lot of random forests with pruned trees; computes the average E_in and E_out; plots the first trained random forest as an example.

## Homework 4
### Question 11 - 14
`hw4q11/hw4q11_handwritten.py`: a handwritten neural network (nnets are so complicated!).

Node: in `/hw4q11/`, use

    python hw4q11_handwritten.py
    
will run all solutions to questions 11 - 14. Be careful, my neural networks are super slow. So, when choosing parameters, you might want to change `nexperiments` to some smaller value first to eliminate some clearly bad candidates.

### Question 15 - 18
`hw4q15/hw4q15.py`: a handwritten KNN

### Question 19, 20
`hw4q19/hw4q19_handwritten.py`: a handwritten kmeans

`hw4q19/hw4q19_sklearn.py`: kmeans using sklearn

Both should be fine.