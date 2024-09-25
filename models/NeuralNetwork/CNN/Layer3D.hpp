#pragma once

#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "../Layer1D.hpp"
#include "../../../utils/vectorOperations.hpp"
#include "../../../utils/logistic.hpp"
#include "../../../utils/activation.hpp"
using Eigen::Matrix;
using namespace std;

template<typename T>
class Layer3D {
private : 
    triplet<size_t, size_t, size_t> _input_dim;
    triplet<size_t, size_t, size_t> _output_dim;
    Layer1D<T> _lay;

public :
    Layer3D() : _input_dim({0, 0, 0}), _output_dim({0, 0, 0}), _lay() {}

    Layer3D(const triplet<size_t, size_t, size_t> &input_dim, const triplet<size_t, size_t, size_t> &output_dim, string activation="sigmoid")
     : _input_dim(input_dim), _output_dim(output_dim), _lay(input_dim.first*input_dim.second*input_dim.third, output_dim.first*output_dim.second*output_dim.third, activation) {}

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getWeights() const {
        return this->_lay.getWeights();
    }

    function<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>)> getActivation() const {
        return get3DFunction<T>(this->_lay.getActivationName());
    }
    
    function<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>)> getActivationPrime() const {
        return get3DFunction<T>(getPrimeFunction(this->_lay.getActivationName()));
    }

    Matrix<T, Eigen::Dynamic, 1> getBias() const {
        return this->_lay.getBias();
    }

    triplet<size_t, size_t, size_t> getInputDim() const {
        return this->_input_dim;
    }

    triplet<size_t, size_t, size_t> getOutputDim() const {
        return this->_output_dim;
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> feedforward(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input) const {
        if(input.size() != _input_dim.first) throw invalid_argument("NeuralNetwork3D::feedforward input.size() != _input_dim.first");
        if(input[0].rows() != _input_dim.second) throw invalid_argument("NeuralNetwork3D::feedforward input.rows() != _input_dim.second");
        if(input[0].cols() != _input_dim.third) throw invalid_argument("NeuralNetwork3D::feedforward input.cols() != _input_dim.third");
        Matrix<T, Eigen::Dynamic, 1> X = to_vector(input);
        return to_matrix(this->_lay.feedforward(to_vector(input)), _output_dim.first, _output_dim.second, _output_dim.third);
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> feedforwardNoApply(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input) const {
        return to_matrix(this->_lay.feedforwardNoApply(to_vector(input)), _output_dim.first, _output_dim.second, _output_dim.third);
    }

    T error(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &X, const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Y) const {
        auto Y_ = this->feedforward(X);
        T error = 0.0;
        for(size_t i = 0; i < _output_dim.first; ++i) 
            for(size_t j = 0; j < _output_dim.second; ++j)
                for(size_t k = 0; k < _output_dim.third; ++k)
                    error += std::pow(Y_[i](j, k) - Y[i](j, k), 2);
        return error;
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> backpropagation(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Z, 
                                                                      const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &delta) const {
        auto Z_vec     = to_vector(Z);
        auto delta_vec = to_vector(delta);
        Matrix<T, Eigen::Dynamic, 1> ret = this->_lay.backpropagation(Z_vec, delta_vec);
        return to_matrix(ret, _input_dim.first, _input_dim.second, _input_dim.third);
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> backpropCNN(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Z,
                                                                  const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &delta,
                                                                  const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input
                                                                  ) const {
        auto Z_vec     = to_vector(Z);
        auto delta_vec = to_vector(delta);
        Matrix<T, Eigen::Dynamic, 1> ret = this->_lay.backpropagation(Z_vec, delta_vec);
        return to_matrix(ret, _input_dim.first, _input_dim.second, _input_dim.third);
    }

    void update(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &delta, 
                const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input,
                T learning_rate) {
        auto delta_vec = to_vector(delta);
        auto input_vec = to_vector(input);
        this->_lay.update(delta_vec, input_vec, learning_rate);
    }

    void updateCNN(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &delta, 
                   const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input,
                   const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &A,
                   T learning_rate) {
        auto delta_vec = to_vector(delta);
        auto input_vec = to_vector(input);
        this->_lay.update(delta_vec, input_vec, learning_rate);
    }

    void print() const {
        auto weight = this->getWeights();
        auto bias   = this->getBias();
        cout << "Weights : " << endl;
        cout << weight << endl;
        cout << "Bias : " << endl;
        cout << bias << endl;
    }
};