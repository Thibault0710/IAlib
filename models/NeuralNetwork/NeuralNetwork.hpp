#pragma once

#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "Layer1D.hpp"
using Eigen::Matrix;

/*
TODO : set a parameter of activation function
*/


template<typename T>
class NeuralNetwork {
private : 
    size_t _input_dim;
    size_t _output_dim;
    std::vector<Layer1D<T>> _net;
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _X;
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _Y;

public : 
    NeuralNetwork(const std::vector<size_t> &tailles) : _input_dim(tailles.front()), _output_dim(tailles.back()), _net(tailles.size() - 1) {
        for(size_t i = 0; i < tailles.size() - 1; ++i) _net[i] = Layer1D<T>(tailles[i], tailles[i+1]); // On garde la sigmoid pour l'instant et on pourra faire d'autres consturecteurs apres
    }

    NeuralNetwork(const vector<Layer1D<T>> &net) : _input_dim(net[0].getInputDim()), _output_dim(net.back().getOutputDim()), _net(net) {}

    std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> getWeights() const {
        std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> weights(_net.size());
        for(size_t i = 0; i < _net.size(); ++i) weights[i] = this->_net[i].getWeights();
        return weights;
    }

    void loadData(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Y) {
        if(X.cols() !=  _input_dim) throw std::invalid_argument("NeuralNetwork::loadData X.cols() sould be equal to _input_dim");
        if(Y.cols() != _output_dim) throw std::invalid_argument("NeuralNetwork::loadData Y.cols() sould be equal to _output_dim");

        _X = X;
        _Y = Y;
    }

    void loadData(const std::vector<std::vector<T>> &X, const std::vector<std::vector<T>> &Y) {
        if(X[0].size() !=  _input_dim) throw std::invalid_argument("NeuralNetwork::loadData X[0].size() sould be equal to _input_dim");
        if(Y[0].size() != _output_dim) throw std::invalid_argument("NeuralNetwork::loadData Y[0].size() sould be equal to _output_dim");

        _X = vectorToEigenMatrix(X);
        _Y = vectorToEigenMatrix(Y);
    }

    Matrix<T, Eigen::Dynamic, 1> feedforward(const Matrix<T, Eigen::Dynamic, 1> &X) {
        Matrix<T, Eigen::Dynamic, 1> Y(X);
        for(size_t i = 0; i < _net.size(); ++i) Y = _net[i].feedforward(Y);
        return Y;
    }

    Matrix<T, Eigen::Dynamic, 1> feedforwardGetWeights(const Matrix<T, Eigen::Dynamic, 1> &X, size_t layer) {// for convolutional networks
        Matrix<T, Eigen::Dynamic, 1> Y(X);
        for(size_t i = 0; i < layer; ++i) Y = _net[i].feedforward(Y);
        return Y;
    }

    Matrix<T, Eigen::Dynamic, 1> feedforward(const std::vector<T> &X) {
        Matrix<T, Eigen::Dynamic, 1> Y(vectorToEigenMatrix(X));
        for(size_t i = 0; i < _net.size(); ++i) Y = _net[i].feedforward(Y);
        return Y;
    }

    Matrix<T, Eigen::Dynamic, 1> feedforwardGetWeights(const std::vector<T> &X, size_t layer) {
        Matrix<T, Eigen::Dynamic, 1> Y(vectorToEigenMatrix(X));
        for(size_t i = 0; i < layer; ++i) Y = _net[i].feedforward(Y);
        return Y;
    }

    double score(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Y) {
        double score = 0.0;
        for(size_t i = 0; i < X.rows(); ++i) {
            Matrix<T, Eigen::Dynamic, 1> prediction = this->feedforward(X.row(i).transpose());
            int maxCoefPred {0}, maxCoefLabel {0};
            prediction.maxCoeff(&maxCoefPred);
            Y.row(i).maxCoeff(&maxCoefLabel);
            score += (maxCoefLabel == maxCoefPred) ? 1 : 0;
        }
        return score / X.rows();
    }

    double score(const std::vector<std::vector<T>> &X, const std::vector<std::vector<T>> &Y) {
        return this->score(vectorToEigenMatrix(X), vectorToEigenMatrix(Y));
    }

    T error(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Y) {
        T ret = 0;
        for(size_t i = 0; i < X.rows(); ++i) ret += (this->feedforward(X.row(i).transpose()) - Y.row(i).transpose()).squaredNorm();
        return ret / static_cast<T>(X.rows());
    }
        /*
        On calcule le gradient par rapport a l'entrée meme sur la premiere couche, c'est inutile mais on en a besoin pour un réseau de neurones convolutionnel,
        on adatera juste l'ecriture
        */
    void fit(size_t epochs = 10, T learning_rate = 0.32, size_t batch_size = 128, double dropout_rate = 0.0) {

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            shuffleRows(_X, _Y);
            
            for (size_t i = 0; i < _X.rows(); i+=batch_size) {
                size_t end = min((int) _X.rows(), (int) (i+batch_size));
                std::vector<Matrix<T, Eigen::Dynamic, 1>> acc_delta(this->_net.size() + 1);
                std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> acc_weight_delta(this->_net.size());

                for(size_t j = 0; j < this->_net.size(); ++j) {
                    acc_delta[j]        = Matrix<T, Eigen::Dynamic, 1>::Zero(this->_net[j].getInputDim());
                    acc_weight_delta[j] = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(_net[j].getOutputDim(), _net[j].getInputDim());
                }
                acc_delta.back() = Matrix<T, Eigen::Dynamic, 1>::Zero(this->_net.back().getOutputDim());

                for(size_t index = i; index < end; ++index) {
                    std::vector<Matrix<T, Eigen::Dynamic, 1>> Z(this->_net.size() + 1);
                    std::vector<Matrix<T, Eigen::Dynamic, 1>> A(this->_net.size() + 1);
                    A[0] = _X.row(index).transpose();
                    Z[0] = _X.row(index).transpose();

                    for (size_t j = 0; j < this->_net.size(); ++j) {
                        Z[j+1]   = this->_net[j].feedforwardNoApply(A[j]);
                        A[j+1]   = _net[j].getActivation()(Z[j+1]);

                        if (dropout_rate > 0.0 && dropout_rate < 1.0) {
                            Matrix<T, Eigen::Dynamic, 1> dropout_mask = Matrix<T, Eigen::Dynamic, 1>::Random(A[j+1].rows()).unaryExpr([&dropout_rate](T x) { return x > dropout_rate ? 1.0 : 0.0; });
                            A[j+1] = A[j+1].array() * dropout_mask.array();
                        }
                    }
                    std::vector<Matrix<T, Eigen::Dynamic, 1>> delta(this->_net.size() + 1);
                    delta.back() = _net.back().getActivationPrime()(Z.back()).array() * (A.back() - _Y.row(index).transpose()).array();
                    for(size_t j = 1; j < this->_net.size()+1; ++j) delta[delta.size() - 1 - j] = _net[_net.size() - j].backpropagation(Z[Z.size()-1-j], delta[delta.size() - j]);

                    // Update of acc_ : 
                    for(size_t j = 0; j < this->_net.size()+1; ++j) {
                        acc_delta[j] += delta[j] / batch_size;
                        if(j != this->_net.size())
                            acc_weight_delta[j] += (delta[j+1] * A[j].transpose())/batch_size;
                    }
                }

                for(size_t j = 0; j < this->_net.size(); ++j) this->_net[j].update_batch(acc_delta[j+1], acc_weight_delta[j], learning_rate);
            }
        }
    }
    
    void SGD(size_t epochs = 10, T learning_rate = 0.01) {

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            shuffleRows(_X, _Y);
            
            for (size_t i = 0; i < _X.rows(); ++i) {
                std::vector<Matrix<T, Eigen::Dynamic, 1>> Z(this->_net.size() + 1);
                std::vector<Matrix<T, Eigen::Dynamic, 1>> A(this->_net.size() + 1);
                A[0] = _X.row(i).transpose();
                Z[0] = _X.row(i).transpose();

                for (size_t j = 0; j < this->_net.size(); ++j) {
                    Z[j+1]   = this->_net[j].feedforwardNoApply(A[j]);
                    A[j+1]   = _net[j].getActivation()(Z[j+1]);
                }

                std::vector<Matrix<T, Eigen::Dynamic, 1>> delta(this->_net.size() + 1);
                delta.back() = _net.back().getActivationPrime()(Z.back()).array() * (A.back() - _Y.row(i).transpose()).array();
                for(size_t j = 1; j < this->_net.size()+1; ++j) delta[delta.size() - 1 - j] = _net[_net.size() - j].backpropagation(Z[Z.size()-1-j], delta[delta.size() - j]);
                for(size_t j = 0; j < this->_net.size(); ++j) _net[j].update(delta[j+1], A[j], learning_rate);
            }
        }
    }
    

    void print() {
        for(size_t i = 0; i < _net.size(); ++i) {
            std::cout << "Layer : "<< i << endl;
            _net[i].print();
        }
    }

};