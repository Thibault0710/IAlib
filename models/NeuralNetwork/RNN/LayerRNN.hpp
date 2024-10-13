#pragma once

#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "../../../utils/vectorOperations.hpp"
#include "../../../utils/logistic.hpp"
#include "../../../utils/activation.hpp"
using Eigen::Matrix;

template<typename T>
class RNN {
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

    RNN() : _input_dim(0), _hidden_dim(0), _output_dim(0), _activation("sigmoid") {}

    RNN(size_t input_dim, size_t hidden_dim, size_t output_dim, string activation = "sigmoid")
     : _input_dim(input_dim), _hidden_dim(hidden_dim), _output_dim(output_dim), _activation(activation) {

        std::random_device rd;
        std::mt19937 gen(rd());
        double mean   = 0.0;
        double stddev = 1.0;
        std::normal_distribution<double> d(mean, stddev);
        auto rand_norm = [&d, &gen]() { return d(gen); };

        _weights_input     = Eigen::MatrixXd::NullaryExpr(_hidden_dim, _input_dim,  rand_norm);
        _weights_hidden    = Eigen::MatrixXd::NullaryExpr(_hidden_dim, _hidden_dim, rand_norm);
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
        next_hidden                                = get1DFunction<T>(_activation)(next_hidden);

        Matrix<T, Eigen::Dynamic, 1> output = _weights_output*next_hidden + _bias_output;
        output                              = get1DFunction<T>(_activation)(output);

        return {next_hidden, output};
    }

    pair<Matrix<T, Eigen::Dynamic, 1>, Matrix<T, Eigen::Dynamic, 1>> feedforward(const Matrix<T, Eigen::Dynamic, 1> &input) const { // Thus we assume hidden is null
        Matrix<T, Dynamic, 1> hidden = Matrix<T, Dynamic, 1>::Zero(_hidden_dim);      
        return this->feedforward(input, hidden);
    }

    Matrix<T, Eigen::Dynamic, 1> feedforwardResult(const Matrix<T, Eigen::Dynamic, 1> &input) const { // Thus we assume hidden is null
        Matrix<T, Dynamic, 1> hidden = Matrix<T, Dynamic, 1>::Zero(_hidden_dim);      
        return this->feedforward(input, hidden).second;
    }

    T error(const Matrix<T, Eigen::Dynamic, 1> &X, const Matrix<T, Eigen::Dynamic, 1> &Y, const Matrix<T, Eigen::Dynamic, 1> &hidden) {
        auto Y_ = this->feedforwardResult(X, hidden);
        return (Y_ - Y).squaredNorm();
    }

    void fit(const vector<Matrix<T, Eigen::Dynamic, 1>> &X_seq, const vector<Matrix<T, Eigen::Dynamic, 1>> &Y_seq, size_t epochs=10, T learning_rate = 0.01) {
        for(size_t epoch = 0; epoch < epochs; ++epoch) {
            Matrix<T, Eigen::Dynamic, 1> previous_hidden = Matrix<T, Eigen::Dynamic, 1>::Zero(_hidden_dim);

            for(size_t t = 0; t < X_seq.size(); ++t) {
                Matrix<T, Eigen::Dynamic, 1> hidden_non_apply = _weights_hidden*previous_hidden + _weights_input*X_seq[t] + _bias_hidden;
                Matrix<T, Eigen::Dynamic, 1>           hidden = get1DFunction<T>(_activation)(hidden_non_apply);

                Matrix<T, Eigen::Dynamic, 1> output_non_apply = _weights_output*hidden + _bias_output;
                Matrix<T, Eigen::Dynamic, 1>           output = get1DFunction<T>(_activation)(output_non_apply);

                Matrix<T, Eigen::Dynamic, 1> error            = output - Y_seq[t];

                Matrix<T, Eigen::Dynamic, 1>              derive_BO = 2*error.array()*get1DFunction<T>(getPrimeFunction(_activation))(output_non_apply).array();
                Matrix<T, Eigen::Dynamic, Eigen::Dynamic> derive_WO = derive_BO * hidden.transpose();

                Matrix<T, Eigen::Dynamic, 1>              derive_BH = (_weights_output.transpose()*derive_BO).array() * get1DFunction<T>(getPrimeFunction(_activation))(hidden_non_apply).array();
                Matrix<T, Eigen::Dynamic, Eigen::Dynamic> derive_WH = derive_BH * previous_hidden.transpose();
                Matrix<T, Eigen::Dynamic, Eigen::Dynamic> derive_WI = derive_BH * X_seq[t].transpose();

                _weights_output  -= learning_rate * derive_WO;
                _bias_output     -= learning_rate * derive_BO;
                _weights_hidden  -= learning_rate * derive_WH;
                _weights_input   -= learning_rate * derive_WI;
                _bias_hidden     -= learning_rate * derive_BH;

                previous_hidden = hidden;
            }
        }
    }

    void print() {
        cout << "Weights : " << this->_weights << endl;
        cout << "Bias : " << this->_bias << endl;
    }

};
