#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "svm.h"


// set_one_vs_all_problem
// ======================
// input a multiclass svm problem, return a one-vs-all svm problem on idx
// input: svm_problem *prob, double idx
// return: svm_problem *
svm_problem *set_one_vs_all_problem(svm_problem *prob, double idx) {
    svm_problem *ret = new svm_problem;
    
    // m = number of training examples
    int m = prob->l;
    ret->l = m;
    
    // y = labels of training examples
    ret->y = new double[m];
    
    // x = features of training examples
    ret->x = new svm_node *[m];
    for (int i = 0; i < m; ++i) {
        ret->y[i] = (prob->y[i] == idx) ? 1.0 : -1.0;
        ret->x[i] = new svm_node[3] {
            {1, prob->x[i][0].value}, // feature1
            {2, prob->x[i][1].value}, // feature2
            {-1, 0.0},                // end of features
        };
    }

    return ret;
}


// free_problem_content
// ====================
// clean up the pointers in an svm_problem
void free_problem_content(svm_problem *prob) {
    delete[] prob->y;
    for (int i = 0; i < prob->l; ++i) {
        delete[] prob->x[i];
    }
    delete[] prob->x;
}


// load_problem
// ============
// load training examples from a given file
svm_problem *load_problem(std::string file_name) {
    std::ifstream is;
    
    is.open(file_name);
    if (!is.is_open()) {
        std::cerr << "open " << file_name << " failed\n" << std::endl;
        return nullptr;
    }
    
    // load features
    std::vector<double> labels;
    std::vector<double> features1;
    std::vector<double> features2;
    double dd, x1, x2;
    std::string str;
    while (std::getline(is, str)) {
        std::istringstream ss(str);
        ss >> dd;
        ss >> x1;
        ss >> x2;
        labels.push_back(dd);
        features1.push_back(x1);
        features2.push_back(x2);
    }
    
    is.close();
    
    // m = number of training examples
    int m = int(labels.size());
    
    svm_problem *prob = new svm_problem;
    prob->l = m;
    
    // y = labels of training examples
    prob->y = new double[m];

    // x = features of training examples
    prob->x = new svm_node *[m];
    
    for (int i = 0; i < m; ++i) {
        prob->y[i] = labels[i];
        prob->x[i] = new svm_node[3] {
            {1, features1[i]},  // feature1
            {2, features2[i]},  // feature2
            {-1, 0.0},          // end of features
        };
    }

    return prob;
}


// hw1q15_param
// ============
// this is the parameters for homework 1 question 15
svm_parameter *hw1q15_param() {
    svm_parameter *param = new svm_parameter;

    param->svm_type = C_SVC;        // C_SVC
    param->kernel_type = LINEAR;    // use linear kernel
    param->degree = 0;              // ignored
    param->gamma = 0.0;             // ignored
    param->coef0 = 0;               // ignored
    param->cache_size = 100.0;      // cache = 100 MB
    param->eps = 0.001;             // tolerance
    param->C = 0.01;                // C
    param->nr_weight = 0;           // no weight
    param->weight_label = nullptr;  // no weight
    param->weight = nullptr;        // no weight
    param->nu = 0.0;                // ignored
    param->p = 0.0;                 // ignored
    param->shrinking = false;       // shrinking heuristic
    param->probability = true;      // predict probability

    return param;
}


// hw1q16_param
// ============
// this is the parameters for homework 1 question 16
svm_parameter *hw1q16_param() {
    svm_parameter *param = new svm_parameter;
    
    // in this problem, we are going to use the polynomial kernel
    // kernel(x1, x2) = (1 + x1'x2) ^ 2
    param->svm_type = C_SVC;         // C_SVC
    param->kernel_type = POLY;       // use polynomial kernel
    param->degree = 2;               // degree of POLY
    param->gamma = 1.0;              // gamma of POLY
    param->coef0 = 1.0;              // coef0 of POLY
    param->cache_size = 100.0;       // cache = 100 MB
    param->eps = 0.001;              // tolerance
    param->C = 0.01;                 // C
    param->nr_weight = 0;            // no weight
    param->weight_label = nullptr;   // no weight
    param->weight = nullptr;         // no weight
    param->nu = 0.0;                 // ignored
    param->p = 0.0;                  // ignored
    param->shrinking = false;        // shrinking heuristic
    param->probability = true;       // predict probability

    return param;
}


int hw1q15() {
    std::string file_name = "features.train";
    std::ostringstream os;

    os << "-----------------HW1Q15-----------------" << std::endl;

    svm_problem *prob = load_problem(file_name);
    if (prob == nullptr) {
        return -1;
    }

    svm_parameter *param = hw1q15_param();

    const char *err;
    if ((err = svm_check_parameter(prob, param)) != nullptr) {
        std::cout << err << std::endl;
        return -1;
    }

    svm_problem *binprob = set_one_vs_all_problem(prob, 0.0);
    svm_model *model = svm_train(binprob, param);

    double sum_alpha = 0.0;
    double coef = 0.0;
    double w1 = 0.0;
    double w2 = 0.0;
    for (int i = 0; i < model->l; ++i) {
        coef = model->sv_coef[0][i];
        w1 += model->SV[i][0].value * coef;
        w2 += model->SV[i][1].value * coef;
        sum_alpha += coef;
    }

    os << "w = (" << w1 << ", " << w2 << ")" << std::endl;

    free_problem_content(binprob);
    delete binprob;
    svm_free_model_content(model);
    delete model;
    delete param;
    free_problem_content(prob);
    delete prob;
    
    os << "----------------------------------------" << std::endl;

    std::cout << os.str() << std::endl;

    return 0;
}

// hw1q16
// ======
int hw1q16() {
    std::string file_name = "features.train";
    std::ostringstream os;
    
    os << "-----------------HW1Q16-----------------" << std::endl;
    
    svm_problem *prob = load_problem(file_name);
    if (prob == nullptr) {
        return -1;
    }
    
    svm_parameter *param = hw1q16_param();
    
    const char *err;
    if ((err = svm_check_parameter(prob, param)) != nullptr) {
        std::cout << err << std::endl;
        return -1;
    }
    
    // one-vs-all
    std::vector<double> idxs { 0.0, 2.0, 4.0, 6.0, 8.0 };
    for (double idx : idxs) {
        svm_problem *curprob = set_one_vs_all_problem(prob, idx);
        svm_model *curmodel = svm_train(curprob, param);

        // how to obtain the alpha's
        // =========================
        // suppose we have k classes (k = 2 here),
        // and the svm has got l support vectors in all.
        //
        // these sv's are from different classification problems.
        // libsvm treats a multi-class svm as k*(k-1)/2 1-vs-1 two-class svms.
        // we can know where each sv belongs from 'SV' and 'nSV'.
        //
        // so now we get sv_coef[k - 1][l].
        // suppose we are looking at the i'th sv (i < l),
        // and we know that it belongs to class c.
        // then there are k-1 two-class svms related to it:
        // 0-vs-c, 1-vs-c, ..., (c-1)-vs-c, c-vs(c+1), ..., c-vs-(k-1)
        // and we have k-1 coefficients for this sv.
        // each coefficient is y * alpha, where y = +1 or -1.
        //
        // coming back to our problem:
        // we get sv_coef[1][l]
        // so we just need to read sv_coef[0][i] for i = 0..(l-1)
        //
        // in homework 1 question 17, we are asked to get the sum of alphas
        // here is the code:
        double sum_alpha = 0.0;
        for (int i = 0; i < curmodel->l; ++i) {
            sum_alpha += std::abs(curmodel->sv_coef[0][i]);
        }

        int nerror = 0;
        int npositive = 0;
        int nnegative = 0;
        for (int i = 0; i < curprob->l; ++i) {
            double pred = svm_predict(curmodel, curprob->x[i]);
            if (pred == 1.0) {
                npositive++;
            } else {
                nnegative++;
            }
            if (pred != curprob->y[i]) {
                nerror++;
            }
        }
        
        os << "----------------------------------------" << std::endl;
        os << "    For label " << idx << ":" << std::endl;
        os << "    sum(alpha) = " << sum_alpha << std::endl;
        os << "    " << npositive << " are predicted 1.0" << std::endl;
        os << "    " << nnegative << " are predicted -1.0" << std::endl;
        os << "    " << nerror << " are wrong" << std::endl;
        os << "----------------------------------------" << std::endl;

        svm_free_model_content(curmodel);
        delete curmodel;
        free_problem_content(curprob);
        delete curprob;
    }
    
    delete param;
    free_problem_content(prob);
    delete prob;

    std::cout << os.str() << std::endl;
    return 0;
}

void empty_print_func(const char *) {}

int main(int argc, const char * argv[]) {
    // svm_set_print_string_function(&empty_print_func);
    hw1q15();
    hw1q16();
}
