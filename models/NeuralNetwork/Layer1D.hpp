#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "../../utils/vectorOperations.hpp"
#include "../../utils/logistic.hpp"
#include "../../utils/activation.hpp"
using Eigen::Matrix;

template<typename T>
class Layer1D {
private : 
    size_t _input_dim;
    size_t _output_dim;
    string _activation; // typically reLu, sigmoid, tanh
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _weights;
    Matrix<T, Eigen::Dynamic, 1> _bias;

public :

    Layer1D() : _input_dim(0), _output_dim(0), _activation("sigmoid") {}

    Layer1D(size_t input_dim, 
            size_t output_dim,
            string activation = "sigmoid")
     : _input_dim(input_dim), _output_dim(output_dim), _activation(activation) {

        std::random_device rd;
        std::mt19937 gen(rd());

        // Initialisation Glorot normale
        double stddev = std::sqrt(2.0 / (_input_dim + _output_dim));
        std::normal_distribution<double> d(0, stddev);
        auto rand_norm = [&d, &gen]() { return d(gen); };

        _weights = Eigen::MatrixXd::NullaryExpr(output_dim, input_dim, rand_norm);
        _bias    = Eigen::MatrixXd::Zero(output_dim, 1);
    }


    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getWeights() const {
        return _weights;
    }

    size_t getInputDim() const {
        return this->_input_dim;
    }

    string getActivationName() const {
        return this->_activation;
    }

    std::function<Matrix<T, Eigen::Dynamic, 1>(Matrix<T, Eigen::Dynamic, 1>)> getActivation() const {
        return get1DFunction<T>(_activation);
    }
    std::function<Matrix<T, Eigen::Dynamic, 1>(Matrix<T, Eigen::Dynamic, 1>)> getActivationPrime() const {
        return get1DFunction<T>(getPrimeFunction(_activation));
    }

    size_t getOutputDim() const {
        return this->_output_dim;
    }

    Matrix<T, Eigen::Dynamic, 1> getBias() const {
        return _bias;
    }

    Matrix<T, Eigen::Dynamic, 1> feedforward(const Matrix<T, Eigen::Dynamic, 1> &input) const {
        Matrix<T, Eigen::Dynamic, 1> Z   = _weights * input + _bias;
        Z = get1DFunction<T>(_activation)(Z);
        return Z;
    }

    Matrix<T, Eigen::Dynamic, 1> feedforwardNoApply(const Matrix<T, Eigen::Dynamic, 1> &input) const {
        Matrix<T, Eigen::Dynamic, 1> Z   = _weights * input + _bias;
        return Z;
    }

    T error(const Matrix<T, Eigen::Dynamic, 1> &X, const Matrix<T, Eigen::Dynamic, 1> &Y) {
        auto Y_ = this->feedforward(X);
        T error = 0.0;
        for(size_t i = 0; i < _output_dim; ++i) error += std::pow(Y_(i, 0) - Y(i, 0), 2);
        return error;
    }

    Matrix<T, Eigen::Dynamic, 1> backpropagation(const Matrix<T, Eigen::Dynamic, 1> &Z, const Matrix<T, Eigen::Dynamic, 1> &delta) const {
        Matrix<T, Eigen::Dynamic, 1> ret = get1DFunction<T>(getPrimeFunction(_activation))(Z).array()*(_weights.transpose()*delta).array();
        return ret;
    }

    Matrix<T, Eigen::Dynamic, 1> backpropInput(const Matrix<T, Eigen::Dynamic, 1> &delta) const {
        Matrix<T, Eigen::Dynamic, 1> ret = _weights.transpose()*delta;
        return ret;
    }

    void update(const Matrix<T, Eigen::Dynamic, 1> &delta, const Matrix<T, Eigen::Dynamic, 1> &input, T learning_rate) {
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> weights_delta = delta * input.transpose();
        this->_bias    -= learning_rate * delta;
        this->_weights -= learning_rate * weights_delta;
    }

    void update_batch(const Matrix<T, Eigen::Dynamic, 1> &delta, const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &weights_delta, T learning_rate) {
        this->_bias    -= learning_rate * delta;
        this->_weights -= learning_rate * weights_delta;
    }

    void print() {
        cout << "Weights : " << this->_weights << endl;
        cout << "Bias : " << this->_bias << endl;
    }

};