#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "../utils/logistic.hpp"
#include "../utils/activation.hpp"
using Eigen::Matrix;

template <typename T>
class LogisticReg{//it uses sigmoid function
private:
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _X;
        Matrix<T, Eigen::Dynamic, 1> _Y;

        size_t _Size;
        size_t _NbeFeatures;

public:
        LogisticReg(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) {
                size_t rows = X.size();
                size_t cols = X[0].size();

                if(rows != Y.size()) throw std::invalid_argument("LogisticReg::LogisticReg(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) : X and Y must have the same number of columns");
                if(rows == 0 || cols == 0) throw std::invalid_argument("LogisticReg::LogisticReg(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) : X must not be empty");

                _X.resize(rows, cols);
                _Y.resize(rows, 1);

                for(size_t i = 0; i < rows; ++i) {
                        for(size_t j = 0; j < cols; ++j) _X(i, j) = X[i][j];
                        _Y(i, 0) = Y[i];
                }

                _Size        = rows;
                _NbeFeatures = cols;
        }

        LogisticReg(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X, Matrix<T, Eigen::Dynamic, 1> Y) : _X(X), _Y(Y) {
                long int rows = X.rows();
                long int cols = X.cols();

                if(rows != Y.rows()) throw std::invalid_argument("LogisticReg::LogisticReg(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) : X and Y must have the same number of columns");
                if(rows == 0 || cols == 0) throw std::invalid_argument("LogisticReg::LogisticReg(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) : X must not be empty");

                _Size        = rows;
                _NbeFeatures = cols;
        }

        size_t getFeaturesNumber() const {
                return _NbeFeatures;
        }

        size_t getDataSize() const {
                return _Size;
        }

        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getData() const {
                return _X;
        }

        Matrix<T, Eigen::Dynamic, 1> getValues() const {
                return _Y;
        }

        Matrix<T, Eigen::Dynamic, 1> fit(size_t iterations = 100000, double eta = 0.01) {//minimize f:A -> || f(XA) - Y ||², based on gradient descent
                std::srand(static_cast<unsigned int>(std::time(nullptr)));
                Matrix<T, Eigen::Dynamic, 1> A = Matrix<T, Eigen::Dynamic, 1>::Random(_NbeFeatures);
                for(size_t l = 0; l < iterations; ++l) {
                        Matrix<T, Eigen::Dynamic, 1> linearPart  = _X * A;
                        Matrix<T, Eigen::Dynamic, 1> predictions = sigmoidVector(linearPart);
                        Matrix<T, Eigen::Dynamic, 1> error       = predictions - _Y;
                        Matrix<T, Eigen::Dynamic, 1> tmp         = sigmoidPrimeVector(linearPart).array() * error.array();
                        Matrix<T, Eigen::Dynamic, 1> gradient    = _X.transpose() * tmp;
                        A -= eta * gradient;
                }
                return A;
        }

        Matrix<T, Eigen::Dynamic, 1> momentum(size_t iterations = 10000, double eta = 0.1, double momentum = 0.1) {
                std::srand((unsigned int) std::time(nullptr));
                Matrix<T, Eigen::Dynamic, 1> A        = Matrix<T, Eigen::Dynamic, 1>::Random(_NbeFeatures);
                Matrix<T, Eigen::Dynamic, 1> velocity = Matrix<T, Eigen::Dynamic, 1>::Random(_NbeFeatures);
                velocity.setZero();

                for(size_t l = 0; l < iterations; ++l) {
                        Matrix<T, Eigen::Dynamic, 1> linearPart  = _X * A;
                        Matrix<T, Eigen::Dynamic, 1> predictions = sigmoidVector(linearPart);
                        Matrix<T, Eigen::Dynamic, 1> error       = predictions - _Y;
                        Matrix<T, Eigen::Dynamic, 1> tmp         = sigmoidPrimeVector(linearPart).array() * error.array();
                        Matrix<T, Eigen::Dynamic, 1> gradient    = _X.transpose() * tmp;

                        velocity = momentum * velocity - eta * gradient;
                        A        = A + velocity;
                }
                return A;
        }

        Matrix<T, Eigen::Dynamic, 1> Nesterov(size_t iterations = 10000, double eta = 0.1, double momentum = 0.1) {
                std::srand((unsigned int) std::time(nullptr));
                Matrix<T, Eigen::Dynamic, 1> A        = Matrix<T, Eigen::Dynamic, 1>::Random(_NbeFeatures);
                Matrix<T, Eigen::Dynamic, 1> velocity = Matrix<T, Eigen::Dynamic, 1>::Random(_NbeFeatures);
                velocity.setZero();

                for(size_t l = 0; l < iterations; ++l) {
                        Matrix<T, Eigen::Dynamic, 1> linearPart  = _X * (A + momentum * velocity);
                        Matrix<T, Eigen::Dynamic, 1> predictions = sigmoidVector(linearPart);
                        Matrix<T, Eigen::Dynamic, 1> error       = predictions - _Y;
                        Matrix<T, Eigen::Dynamic, 1> tmp         = sigmoidPrimeVector(linearPart).array() * error.array();
                        Matrix<T, Eigen::Dynamic, 1> gradient    = _X.transpose() * tmp;

                        velocity = momentum * velocity - eta * gradient;
                        A        = A + velocity;
                }
                return A;
        }

        Matrix<T, Eigen::Dynamic, 1> fitAffine(size_t iterations = 100000, double eta = 0.01) {//minimize f:A -> || f(XA + B) - Y ||²
                Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X = _X;
                X.conservativeResize(_Size, _NbeFeatures + 1);
                X.col(_NbeFeatures).setOnes();
                LogisticReg<T> Affine(X, _Y);
                return Affine.fit(iterations, eta);
        }

        Matrix<T, Eigen::Dynamic, 1> fitAffineMomentum(size_t iterations = 100000, double eta = 0.001, double momentum = 0.1) {//minimize f:A -> || f(XA + B) - Y ||²
                Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X = _X;
                X.conservativeResize(_Size, _NbeFeatures + 1);
                X.col(_NbeFeatures).setOnes();
                LogisticReg<T> Affine(X, _Y);
                return Affine.momentum(iterations, eta, momentum);
        }

        Matrix<T, Eigen::Dynamic, 1> fitAffineMomentumNesterov(size_t iterations = 100000, double eta = 0.001, double momentum = 0.1) {//minimize f:A -> || f(XA + B) - Y ||²
                Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X = _X;
                X.conservativeResize(_Size, _NbeFeatures + 1);
                X.col(_NbeFeatures).setOnes();
                LogisticReg<T> Affine(X, _Y);
                return Affine.Nesterov(iterations, eta, momentum);
        }
};
