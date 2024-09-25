#pragma once
#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>

using namespace std;
using Eigen::Matrix;

template <typename T>
T sigmoidFunction(T element) {
    return 1 / (1 + exp(-element));
}

template <typename T>
T sigmoidPrimeFunction(T element) {
    T x = sigmoidFunction(element);
    return x * (1 - x);
}

template <typename T>
T identiteFunction(T element) {
    return element;
}

template <typename T>
T identitePrimeFunction(T element) {
    return T(1);
}

template <typename T>
T reLuFunction(T element) {
    return std::max(static_cast<T>(0), element);
}

template <typename T>
T reLuPrimeFunction(T element) {
    return (element > 0) ? static_cast<T>(1) : static_cast<T>(0);
}

template<typename T>
T tanHFunction(T element) {
    return std::tanh(element);
}

template<typename T>
T tanHPrimeFunction(T element) {
    return 1 - pow(tanHFunction(element), 2);
}

template<typename T>
T softplusFunction(T element) {
    return log(1 + exp(element));
}

template<typename T>
T softplusPrimeFunction(T element) {
    return sigmoidFunction<T>(element);
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> sigmoid(Matrix<T, Eigen::Dynamic, 1> X) {
    return X.unaryExpr(&sigmoidFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> sigmoidPrime(Matrix<T, Eigen::Dynamic, 1> X) {
    return X.unaryExpr(&sigmoidPrimeFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> identite(Matrix<T, Eigen::Dynamic, 1> X) {
    return X.unaryExpr(&identiteFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> identitePrime(Matrix<T, Eigen::Dynamic, 1> X) {
    return X.unaryExpr(&identitePrimeFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> reLu(Matrix<T, Eigen::Dynamic, 1> X) {
    return X.unaryExpr(&reLuFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> reLuPrime(Matrix<T, Eigen::Dynamic, 1> X) {
    auto tmp = X.unaryExpr(&reLuPrimeFunction<T>);
    return tmp;
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> tanH(Matrix<T, Eigen::Dynamic, 1> X) {
    return X.unaryExpr(&tanHFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> tanHPrime(Matrix<T, Eigen::Dynamic, 1> X) {
    return X.unaryExpr(&tanHPrimeFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> softplus(const Matrix<T, Eigen::Dynamic, 1> &X) {
    return X.unaryExpr(&softplusFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> softplusPrime(const Matrix<T, Eigen::Dynamic, 1> &X) {
    return X.unaryExpr(&softplusPrimeFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> softmax(const Matrix<T, Eigen::Dynamic, 1> &X) {
    Matrix<T, Eigen::Dynamic, 1> expX = X.array().exp();
    T sum = expX.sum();
    if(sum == 0.0) throw invalid_argument("Softmax merde");
    return expX / sum;
}

template<typename T>
Matrix<T, Eigen::Dynamic, 1> softmaxPrime(const Matrix<T, Eigen::Dynamic, 1> &X) {
    Matrix<T, Eigen::Dynamic, 1> ret = Matrix<T, Eigen::Dynamic, 1>::Ones(X.rows(), 1);
    return ret;
}

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> sigmoid2D(Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X) {
    return X.unaryExpr(&sigmoidFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> sigmoidPrime2D(Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X) {
    return X.unaryExpr(&sigmoidPrimeFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> identite2D(Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X) {
    return X.unaryExpr(&identiteFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> identitePrime2D(Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X) {
    return X.unaryExpr(&identitePrimeFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> reLu2D(Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X) {
    return X.unaryExpr(&reLuFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> reLuPrime2D(Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X) {
    return X.unaryExpr(&reLuPrimeFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tanH2D(Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X) {
    return X.unaryExpr(&tanHFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> tanHPrime2D(Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X) {
    return X.unaryExpr(&tanHPrimeFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> softplus2D(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X) {
    return X.unaryExpr(&softplusFunction<T>);
}

template<typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> softplusPrime2D(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X) {
    return X.unaryExpr(&softplusPrimeFunction<T>);
}

using Eigen::Dynamic;   

template<typename T>
vector<Matrix<T, Dynamic, Dynamic>> sigmoid3D(const vector<Matrix<T, Dynamic, Dynamic>>& X) {
    vector<Matrix<T, Dynamic, Dynamic>> result(X.size());
    for(size_t i = 0; i < X.size(); ++i) result[i] = X[i].unaryExpr(&sigmoidFunction<T>);
    return result;
}


template<typename T>
vector<Matrix<T, Dynamic, Dynamic>> sigmoidPrime3D(const vector<Matrix<T, Dynamic, Dynamic>>& X) {
    vector<Matrix<T, Dynamic, Dynamic>> result(X.size());
    for(size_t i = 0; i < X.size(); ++i) result[i] = X[i].unaryExpr(&sigmoidPrimeFunction<T>);
    return result;
}

template<typename T>
vector<Matrix<T, Dynamic, Dynamic>> identite3D(const vector<Matrix<T, Dynamic, Dynamic>> &X) {
    vector<Matrix<T, Dynamic, Dynamic>> result(X.size());
    for(size_t i = 0; i < X.size(); ++i) result[i] = X[i].unaryExpr(&identiteFunction<T>);
    return result;}

template<typename T>
vector<Matrix<T, Dynamic, Dynamic>> identitePrime3D(const vector<Matrix<T, Dynamic, Dynamic>> &X) {
    vector<Matrix<T, Dynamic, Dynamic>> result(X.size());
    for(size_t i = 0; i < X.size(); ++i) result[i] = X[i].unaryExpr(&identitePrimeFunction<T>);
    return result;
}

template<typename T>
vector<Matrix<T, Dynamic, Dynamic>> reLu3D(const vector<Matrix<T, Dynamic, Dynamic>>& X) {
    vector<Matrix<T, Dynamic, Dynamic>> result(X.size());
    for(size_t i = 0; i < X.size(); ++i) result[i] = X[i].unaryExpr(&reLuFunction<T>);
    return result;
}

template<typename T>
vector<Matrix<T, Dynamic, Dynamic>> reLuPrime3D(const vector<Matrix<T, Dynamic, Dynamic>>& X) {
    vector<Matrix<T, Dynamic, Dynamic>> result(X.size());
    for(size_t i = 0; i < X.size(); ++i) result[i] = X[i].unaryExpr(&reLuPrimeFunction<T>);
    return result;
}

template<typename T>
vector<Matrix<T, Dynamic, Dynamic>> tanH3D(const vector<Matrix<T, Dynamic, Dynamic>>& X) {
    vector<Matrix<T, Dynamic, Dynamic>> result(X.size());
    for(size_t i = 0; i < X.size(); ++i) result[i] = X[i].unaryExpr(&tanHFunction<T>);
    return result;
}

template<typename T>
vector<Matrix<T, Dynamic, Dynamic>> tanHPrime3D(const vector<Matrix<T, Dynamic, Dynamic>>& X) {
    vector<Matrix<T, Dynamic, Dynamic>> result(X.size());
    for(size_t i = 0; i < X.size(); ++i) result[i] = X[i].unaryExpr(&tanHPrimeFunction<T>);
    return result;
}

template<typename T>
vector<Matrix<T, Dynamic, Dynamic>> softplus3D(const vector<Matrix<T, Dynamic, Dynamic>>& X) {
    vector<Matrix<T, Dynamic, Dynamic>> result(X.size());
    for(size_t i = 0; i < X.size(); ++i) result[i] = X[i].unaryExpr(&softplusFunction<T>);
    return result;
}

template<typename T>
vector<Matrix<T, Dynamic, Dynamic>> softplusPrime3D(const vector<Matrix<T, Dynamic, Dynamic>>& X) {
    vector<Matrix<T, Dynamic, Dynamic>> result(X.size()); 
    for(size_t i = 0; i < X.size(); ++i) result[i] = X[i].unaryExpr(&softplusPrimeFunction<T>); 
    return result;
}

template<typename T>
using Func1D = std::function<Matrix<T, Dynamic, 1>(const Matrix<T, Dynamic, 1>&)>;

template<typename T>
using Func2D = std::function<Matrix<T, Dynamic, Dynamic>(const Matrix<T, Dynamic, Dynamic>&)>;

template<typename T>
using Func3D = function<vector<Matrix<T, Dynamic, Dynamic>>(const vector<Matrix<T, Dynamic, Dynamic>>&)>;

template<typename T>
static std::unordered_map<std::string, Func1D<T>> functionMap1D = {
    {"sigmoid",       sigmoid<T>},
    {"sigmoidPrime",  sigmoidPrime<T>},
    {"identite",      identite<T>},
    {"identitePrime", identitePrime<T>},
    {"reLu",          reLu<T>},
    {"reLuPrime",     reLuPrime<T>},
    {"tanH",          tanH<T>},
    {"tanHPrime",     tanHPrime<T>},
    {"softplus",      softplus<T>},
    {"softplusPrime", softplusPrime<T>},
    {"softmax",       softmax<T>},
    {"softmaxPrime",  softmaxPrime<T>}
};

template<typename T>
static std::unordered_map<std::string, Func2D<T>> functionMap2D = {
    {"sigmoid",       sigmoid2D<T>},
    {"sigmoidPrime",  sigmoidPrime2D<T>},
    {"identite",      identite2D<T>},
    {"identitePrime", identitePrime2D<T>},
    {"reLu",          reLu2D<T>},
    {"reLuPrime",     reLuPrime2D<T>},
    {"tanH",          tanH2D<T>},
    {"tanHPrime",     tanHPrime2D<T>},
    {"softplus",      softplus2D<T>},
    {"softplusPrime", softplusPrime2D<T>}
};

template<typename T>
static unordered_map<string, Func3D<T>> functionMap3D = {
    {"sigmoid",       sigmoid3D<T>},
    {"identite",      identite3D<T>},
    {"identitePrime", identitePrime3D<T>},
    {"sigmoidPrime",  sigmoidPrime3D<T>},
    {"reLu",          reLu3D<T>},
    {"reLuPrime",     reLuPrime3D<T>},
    {"tanH",          tanH3D<T>},
    {"tanHPrime",     tanHPrime3D<T>},
    {"softplus",      softplus3D<T>},
    {"softplusPrime", softplusPrime3D<T>}
};

static std::unordered_map<std::string, std::string> functionPrime = {
    {"sigmoid",  "sigmoidPrime"},
    {"identite", "identitePrime"},
    {"reLu",     "reLuPrime"},
    {"tanH",     "tanHPrime"},
    {"softplus", "softplusPrime"},
    {"softmax",  "softmaxPrime"}
};

string getPrimeFunction(const std::string& funcName) {
    auto it = functionPrime.find(funcName);
    if (it != functionPrime.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Function not found: " + funcName);
    }
}

template<typename T>
Func1D<T> get1DFunction(const std::string& funcName) {
    auto it = functionMap1D<T>.find(funcName);
    if (it != functionMap1D<T>.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Function not found: " + funcName);
    }
}

template<typename T>
Func2D<T> get2DFunction(const std::string& funcName) {
    auto it = functionMap2D<T>.find(funcName);
    if (it != functionMap2D<T>.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Function not found: " + funcName);
    }
}

template<typename T>
Func3D<T> get3DFunction(const string& funcName) {
    auto it = functionMap3D<T>.find(funcName);
    if(it != functionMap3D<T>.end())
        return it->second;
    else
        throw invalid_argument("Function not found: " + funcName);
}