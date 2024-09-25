#pragma once

#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "Layer1D.hpp"
#include "../../utils/vectorOperations.hpp"
#include "../../utils/logistic.hpp"
#include "../../utils/activation.hpp"
using Eigen::Matrix;
using namespace std;

template<typename T>
class Layer2D {
private : 
    pair<size_t, size_t> _input_dim;
    pair<size_t, size_t> _output_dim;
    Layer1D<T> _lay;

public :

    Layer2D() : _input_dim({0, 0}), _output_dim({0, 0}), _lay() {}

    Layer2D(const pair<size_t, size_t> &input_dim,
            const pair<size_t, size_t> &output_dim,
            string activation = "sigmoid")
     : _input_dim(input_dim), _output_dim(output_dim), _lay(input_dim.first*input_dim.second, output_dim.first*output_dim.second, activation) {}

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getWeights() const {
        return this->_lay.getWeights();
    }

    Matrix<T, Eigen::Dynamic, 1> getBias() const {
        return this->_lay.getBias();
    }

    pair<size_t, size_t> getInputDim() const {
        return this->_input_dim;
    }

    std::function<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(Matrix<T, Eigen::Dynamic, Eigen::Dynamic>)> getActivation() const {
        return get2DFunction<T>(this->_lay.getActivationName());
    }
    
    function<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(Matrix<T, Eigen::Dynamic, Eigen::Dynamic>)> getActivationPrime() const {
        return get2DFunction<T>(getPrimeFunction(this->_lay.getActivationName()));
    }

    pair<size_t, size_t> getOutputDim() const {
        return this->_output_dim;
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> feedforward(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &input) const {
        Matrix<T, Eigen::Dynamic, 1> input_resize(input.cols() * input.rows());
        for(size_t i = 0; i < input.rows(); ++i) for(size_t j = 0; j < input.cols(); ++j) input_resize(i*input.cols() + j) = input(i, j);
        Matrix<T, Eigen::Dynamic, 1> output = this->_lay.feedforward(input_resize);

        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ret(this->_output_dim.first, this->_output_dim.second);
        for(size_t i = 0; i < _output_dim.first; ++i) for(size_t j = 0; j < _output_dim.second; ++j) ret(i, j) = output(i*_output_dim.second + j);
        return ret;
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> feedforward(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input) const {
        vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> output(input.size());
        transform(input.begin(), input.end(), output.begin(), [this](Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat){ return this->feedforward(mat); });
        return output;
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> feedforwardNoApply(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &input) const {
        Matrix<T, Eigen::Dynamic, 1> input_resize(input.cols() * input.rows());
        for(size_t i = 0; i < input.rows(); ++i) for(size_t j = 0; j < input.cols(); ++j) input_resize(i*input.cols() + j) = input(i, j);
        Matrix<T, Eigen::Dynamic, 1> output = this->_lay.feedforwardNoApply(input_resize);

        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ret(this->_output_dim.first, this->_output_dim.second);
        for(size_t i = 0; i < _output_dim.first; ++i) for(size_t j = 0; j < _output_dim.second; ++j) ret(i, j) = output(i*_output_dim.second + j);
        return ret;
    }
    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> feedforwardNoApply(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input) const {
        vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> output(input.size());
        transform(input.begin(), input.end(), output.begin(), [this](Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat){ return this->feedforwardNoApply(mat); });
        return output;
    }

    T error(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Y) const {
        auto Y_ = this->feedforward(X);
        T error = 0.0;
        for(size_t i = 0; i < _output_dim.first; ++i) for(size_t j = 0; j < _output_dim.second; ++j) error += std::pow(Y_(i, j) - Y(i, j), 2);
        return error;
    }

    T error(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &X, const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Y) const {
        auto Y_ = this->feedforward(X);
        T error = 0.0;
        for(size_t k = 0; k < Y_.size(); ++k) for(size_t i = 0; i < _output_dim.first; ++i) for(size_t j = 0; j < _output_dim.second; ++j) error += std::pow(Y_[k](i, j) - Y[k](i, j), 2);
        return error;
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> backpropagation(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Z, const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &delta) const {
        auto Z_vec     = to_vector(Z);
        auto delta_vec = to_vector(delta);
        Matrix<T, Eigen::Dynamic, 1> ret = this->_lay.backpropagation(Z_vec, delta_vec);
        return to_matrix(ret, _input_dim.first, _input_dim.second);
    }

    void update(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &delta, const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &input, T learning_rate) {
        auto delta_vec = to_vector(delta);
        auto input_vec = to_vector(input);
        this->_lay.update(delta_vec, input_vec, learning_rate);
    }

    void print() const {
        cout << "Weights : " << this->getWeights() << endl;
        cout << "Bias : " << this->getBias() << endl;
    }
};