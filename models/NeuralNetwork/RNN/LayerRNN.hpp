#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "../../../utils/vectorOperations.hpp"
#include "../../../utils/logistic.hpp"
#include "../../../utils/activation.hpp"
using Eigen::Matrix;

template<typename T>
class LayerRNN {
private : 
    size_t _input_dim;
    size_t _hidden_dim;
    size_t _output_dim;
    string _activation;
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _weights_input;
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _weights_hidden;
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _weights_output;
    Matrix<T, Eigen::Dynamic, 1> _bias_hidden;
    Matrix<T, Eigen::Dynamic, 1> _bias_output;

public :

    LayerRNN() : _input_dim(0), _hidden_dim(0), _output_dim(0), _activation("sigmoid") {}

    LayerRNN(size_t input_dim, size_t hidden_dim, size_t output_dim, string activation = "sigmoid")
     : _input_dim(input_dim), _hidden_dim(hidden_dim), _output_dim(output_dim), _activation(activation) {

        std::random_device rd;
        std::mt19937 gen(rd());
        double mean   = 0.0;
        double stddev = 1.0;
        std::normal_distribution<double> d(mean, stddev);
        auto rand_norm = [&d, &gen]() { return d(gen); };

        _weights_input     = Eigen::MatrixXd::NullaryExpr(_hidden_dim, _input_dim,  rand_norm);
        _weights_hidden = Eigen::MatrixXd::NullaryExpr(_hidden_dim, _hidden_dim, rand_norm);
        _weights_output    = Eigen::MatrixXd::NullaryExpr(_output_dim, _hidden_dim, rand_norm);
        _bias_hidden       = Eigen::MatrixXd::NullaryExpr(_hidden_dim, 1,           rand_norm);
        _bias_output       = Eigen::MatrixXd::NullaryExpr(_output_dim, 1,           rand_norm);
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getWeightsHidden() const {
        return _weights_hidden;
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getWeightsOutput() const {
        return _weights_output;
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getWeightsInput() const {
        return _weights_input;
    }

    size_t getInputDim() const {
        return this->_input_dim;
    }

    size_t getOutputDim() const {
        return this->_output_dim;
    }

    size_t getHiddenDim() const {
        return this->_hidden_dim;
    }

    Matrix<T, Eigen::Dynamic, 1> getBiasOutput() const {
        return _bias_output;
    }

    Matrix<T, Eigen::Dynamic, 1> getBiasHidden() const {
        return _bias_hidden;
    }

    pair<Matrix<T, Eigen::Dynamic, 1>, Matrix<T, Eigen::Dynamic, 1>> feedforward(const Matrix<T, Eigen::Dynamic, 1> &input, const Matrix<T, Eigen::Dynamic, 1> &hidden) const {
        //first is the next activation and second is the output
        Matrix<T, Eigen::Dynamic, 1> next_hidden   = _weights_input*input + _weights_hidden*hidden + _bias_hidden;
        next_hidden = next_hidden.unaryExpr(_activation);

        Matrix<T, Eigen::Dynamic, 1> output = _weights_output*next_hidden + _bias_output;
        output = output.unaryExpr(_activation);

        return {next_hidden, output};
    }

    Matrix<T, Eigen::Dynamic, 1> feedforward(const Matrix<T, Eigen::Dynamic, 1> &input) const { // Thus we assume hidden is null
        Matrix<T, Dynamic, 1> hidden = Matrix<T, Dynamic, 1>::Zero(_hidden_dim);      
        return this->feedforward(input, hidden);
    }

    T error(const Matrix<T, Eigen::Dynamic, 1> &X, const Matrix<T, Eigen::Dynamic, 1> &Y, const Matrix<T, Eigen::Dynamic, 1> &hidden) {
        auto Y_ = this->feedforward(X, hidden);
        return (Y_ - Y).squaredNorm();
    }
/*
    Matrix<T, Eigen::Dynamic, 1> backpropagation(const Matrix<T, Eigen::Dynamic, 1> &Z, const Matrix<T, Eigen::Dynamic, 1> &delta) const {
        Matrix<T, Eigen::Dynamic, 1> ret = (Z.unaryExpr([this](T t){ return _activation_prime(t); }).array())*(_weights.transpose()*delta).array();
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
*/  
    void print() {
        cout << "Weights : " << this->_weights << endl;
        cout << "Bias : " << this->_bias << endl;
    }

};