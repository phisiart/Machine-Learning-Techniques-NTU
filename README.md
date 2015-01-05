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
