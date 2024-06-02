#include <math.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
using Eigen::Matrix;

template <typename T>
T sigmoid(double element) {
        return 1/(1+std::exp(-element));
}

template <typename T>
T sigmoidPrime(T element) {
        T x = std::exp(-element);
        return x / ((1+x)*(1+x));
}

template <typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> sigmoidMatrix(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat) {
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ret(mat.rows(), mat.cols());
        for(int i = 0; i < mat.rows(); ++i) for(int j = 0; j < mat.cols(); ++j) ret(i, j) = sigmoid<T>(mat(i, j));
        return ret;
}

template <typename T>
Matrix<T, Eigen::Dynamic, 1> sigmoidVector(const Matrix<T, Eigen::Dynamic, 1> &mat) {
        Matrix<T, Eigen::Dynamic, 1> ret(mat.rows(), mat.cols());
        for(int i = 0; i < mat.rows(); ++i) for(int j = 0; j < mat.cols(); ++j) ret(i, j) = sigmoid<T>(mat(i, j));
        return ret;
}

template <typename T>
Matrix<T, Eigen::Dynamic, Eigen::Dynamic> sigmoidPrimeMatrix(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat) {
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ret(mat.rows(), mat.cols());
        for(int i = 0; i < mat.rows(); ++i) for(int j = 0; j < mat.cols(); ++j) ret(i, j) = sigmoidPrime<T>(mat(i, j));
        return ret;
}

template <typename T>
Matrix<T, Eigen::Dynamic, 1> sigmoidPrimeVector(const Matrix<T, Eigen::Dynamic, 1> &mat) {
        Matrix<T, Eigen::Dynamic, 1> ret(mat.rows(), mat.cols());
        for(int i = 0; i < mat.rows(); ++i) for(int j = 0; j < mat.cols(); ++j) ret(i, j) = sigmoidPrime<T>(mat(i, j));
        return ret;
}
