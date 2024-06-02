#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
using Eigen::Matrix;

template <typename T>
class RegLin{
private:
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _X;//the data collected
        Matrix<T, Eigen::Dynamic, 1> _Y;//the associated result

        Matrix<T, Eigen::Dynamic, 1> _Coefs; //the coefficient that we compute

        size_t _Size;//the number of data
        size_t _NbeFeatures;// the size of of data

public:
        RegLin(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) {
                size_t rows = X.size();
                size_t cols = X[0].size();

                if(rows != Y.size()) throw std::invalid_argument("RegLin::RegLin(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) : X and Y must have the same number of columns");
                if(rows == 0 || cols == 0) throw std::invalid_argument("RegLin::RegLin(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) : X must not be empty");

                _X.resize(rows, cols);
                _Y.resize(rows, 1);
                _Coefs.resize(cols, 1);

                for(size_t i = 0; i < rows; ++i) {
                        for(size_t j = 0; j < cols; ++j) _X(i, j) = X[i][j];
                        _Y(i, 0) = Y[i];
                }

                _Size        = rows;
                _NbeFeatures = cols;
        }

        RegLin(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X, Matrix<T, Eigen::Dynamic, 1> Y) : _X(X), _Y(Y) {
                long int rows = X.rows();
                long int cols = X.cols();

                if(rows != Y.rows()) throw std::invalid_argument("RegLin::RegLin(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) : X and Y must have the same number of columns");
                if(rows == 0 || cols == 0) throw std::invalid_argument("RegLin::RegLin(const std::vector<std::vector<T>> &X, const std::vector<T> &Y) : X must not be empty");

                _Coefs.resize(cols, 1);

                _Size        = rows;
                _NbeFeatures = cols;
        }

        Matrix<T, Eigen::Dynamic, 1> fit() {//minimize  A -> || AX - Y ||²
                Matrix<T, Eigen::Dynamic, 1> coefficients = (_X.transpose() * _X).ldlt().solve(_X.transpose() * _Y);
                _Coefs = coefficients;
                return coefficients;
        }

        Matrix<T, Eigen::Dynamic, 1> fitAffine() {// last coefficient represents the constant value we add, try to minimize A -> || AX + b - Y ||² where b is the last coefficient of the returning value
                Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X = _X;
                X.conservativeResize(_Size, _NbeFeatures + 1);
                X.col(_NbeFeatures).setOnes();
                RegLin<T> Affine(X, _Y);
                return Affine.fit();
        }

        size_t getFeaturesNumber() const {
                return _NbeFeatures;
        }

        size_t getDataSize() const {
                return _Size;
        }

        Matrix<T, Eigen::Dynamic, 1> getCoefs() const {
                return _Coefs;
        }

        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getData() const {
                return _X;
        }

        Matrix<T, Eigen::Dynamic, 1> getValues() const {
                return _Y;
        }


        T feedforward(const std::vector<T> &X) {
                if(X.size() != _Coefs.rows()) throw std::invalid_argument("LinearRegression::feedforward X.rows() != _Coefs.rows()");
                T ret = 0;

                for(int i = 0; i < X.size(); ++i) ret += X[i] * _Coefs[i];
                return ret;
        }

        Matrix<T, Eigen::Dynamic, 1> feedforward(const std::vector<std::vector<T>> &X) {
                Matrix<T, Eigen::Dynamic, 1> ret;
                ret.resize(X.size());

                for(size_t i = 0; i < X.size(); ++i) ret[i] = this->feedforward(X[i]);
                return ret;
        }
};
