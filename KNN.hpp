#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "./utils/vectorOperations.hpp"
using Eigen::Matrix;

template<typename T>
class KNN {
private :
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _X; // The Data used to train the model
    Matrix<int, Eigen::Dynamic, 1> _Y; // The labels of the data, we suppose they are modeled by integer
    size_t _K; // The number of neighboors we have
    
public :
    KNN(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, const Matrix<int, Eigen::Dynamic, 1> &Y, size_t K) : _X(X), _Y(Y), _K(K) {
    }

    KNN(const std::vector<std::vector<T>> &X, const std::vector<int> &Y, size_t K) {
        size_t rows = X.size();
        size_t cols = X[0].size();

        if(rows != Y.size()) throw std::invalid_argument("RegLin::RegLin(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) : X and Y must have the same number of columns");
        if(rows == 0 || cols == 0) throw std::invalid_argument("RegLin::RegLin(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) : X must not be empty");

        _X.resize(rows, cols);
        _Y.resize(rows, 1);

        for(size_t i = 0; i < rows; ++i) {
            for(size_t j = 0; j < cols; ++j) _X(i, j) = X[i][j];
            _Y(i, 0) = Y[i];
        }

        _K = K;
    }

    Matrix<int, Eigen::Dynamic, 1> fit(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Xtest) {// The data we want to label using KNN algorithm
        Matrix<int, Eigen::Dynamic, 1> labels(Xtest.rows());

        for(size_t i = 0; i < Xtest.rows(); ++i) {
            std::vector<int> nearestLabels(_K, -1);
            std::vector<double> nearestNorms(_K, std::numeric_limits<double>::infinity());

            for(size_t j = 0; j < _X.rows(); ++j) {
                double norm = (Xtest.row(i) - _X.row(j)).norm();
                size_t k = 0;
                while(norm > nearestNorms[k] && ++k < _K);

                if(k != _K) {
                    nearestLabels[k] = _Y(j,0);
                    nearestNorms[k]  = norm;
                }
            }
            //now we have the nearest neighbors

            labels(i, 0) = mostFrequentValue(nearestLabels);
        }

        return labels;
    }

    Matrix<int, Eigen::Dynamic, 1> fit(const std::vector<std::vector<T>> &Xtest) {
        return this->fit(vectorToEigenMatrix(Xtest));
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getData() {
        return _X;
    }

};