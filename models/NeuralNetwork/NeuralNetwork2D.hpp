#pragma once
#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "Layer2D.hpp"
using Eigen::Matrix;

/*
TODO : set a parameter of activation function
*/


template<typename T>
class NeuralNetwork2D {
private : 
    pair<size_t, size_t> _input_dim;
    pair<size_t, size_t> _output_dim;
    std::vector<Layer2D<T>> _net;
    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _X;
    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _Y;

public : 
    NeuralNetwork2D(const std::vector<pair<size_t, size_t>> &tailles) : _input_dim(tailles.front()), _output_dim(tailles.back()), _net(tailles.size() - 1) {
        for(size_t i = 0; i < tailles.size() - 1; ++i) _net[i] = Layer2D<T>(tailles[i], tailles[i+1]); // On garde la sigmoid pour l'instant et on pourra faire d'autres consturecteurs apres
    }

    NeuralNetwork2D(const vector<Layer2D<T>> &net) : _input_dim(net[0].getInputDim()), _output_dim(net.back().getOutputDim()), _net(net) {}

    std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> getWeights() const {
        std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> weights(_net.size());
        for(size_t i = 0; i < _net.size(); ++i) weights[i] = this->_net[i].getWeights();
        return weights;
    }

    void loadData(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &X, const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Y) {
        _X = X;
        _Y = Y;
    }

    void loadData(const vector<std::vector<std::vector<T>>> &X, const vector<std::vector<std::vector<T>>> &Y) {
        _X = vectorToEigenMatrix(X);
        _Y = vectorToEigenMatrix(Y);
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> feedforward(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X) {
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y(X);
        for(size_t i = 0; i < _net.size(); ++i) Y = _net[i].feedforward(Y);
        return Y;
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> feedforwardPrint(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X) {
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y(X);
        for(size_t i = 0; i < _net.size(); ++i) {
            cout << "Layer: " << i << endl;
            cout << Y << endl;
            Y = _net[i].feedforward(Y);
        }
        cout << "Last layer: " << endl;
        cout << Y <<endl;
        return Y;
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> feedforward(const std::vector<std::vector<T>> &X) {
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y(vectorToEigenMatrix(X));
        for(size_t i = 0; i < _net.size(); ++i) Y = _net[i].feedforward(Y);
        return Y;
    }

    double score(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& X, const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>& Y) {
        double score = 0.0;
        for(size_t i = 0; i < X.size(); ++i) {
            Matrix<T, Eigen::Dynamic, Eigen::Dynamic> prediction = this->feedforward(X[i]);
            
            int maxIndexPred = 0, maxIndexLabel = 0;
            T maxValPred = prediction(0, 0), maxValLabel = Y[i](0, 0);

            for (int row = 0; row < prediction.rows(); ++row) {
                for (int col = 0; col < prediction.cols(); ++col) {
                    if (prediction(row, col) > maxValPred) {
                        maxValPred = prediction(row, col);
                        maxIndexPred = row * prediction.cols() + col;
                    }
                    if (Y[i](row, col) > maxValLabel) {
                        maxValLabel = Y[i](row, col);
                        maxIndexLabel = row * Y[i].cols() + col;
                    }
                }
            }
            score += (maxIndexLabel == maxIndexPred) ? 1.0 : 0.0;
        }
        return score / X.size();
    }

    double score(const std::vector<std::vector<std::vector<T>>> &X, const std::vector<std::vector<std::vector<T>>> &Y) {
        return this->score(vectorToEigenMatrix(X), vectorToEigenMatrix(Y));
    }

    T error(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &X, const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Y) {
        T ret = 0;
        for(size_t i = 0; i < X.size(); ++i) ret += (this->feedforward(X[i]) - Y[i]).squaredNorm();
        return ret / static_cast<T>(X.size());
    }

    T error(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Y) {
        return (this->feedforward(X) - Y).squaredNorm();
    }

    void fit(size_t epochs = 10, T learning_rate = 0.01) {
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            shuffleRows(_X, _Y);
            
            for (size_t i = 0; i < _X.size(); ++i) {
                // Passe avant
                std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> Z(this->_net.size() + 1);
                std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> A(this->_net.size() + 1);
                A[0] = _X[i];
                Z[0] = A[0];

                for (size_t j = 0; j < this->_net.size(); ++j) {
                    Z[j+1]   = this->_net[j].feedforwardNoApply(A[j]);
                    A[j+1]   = _net[j].getActivation()(Z[j+1]);  
                }
                //Passe arrière
                std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> delta(this->_net.size() + 1);
                delta.back() = _net.back().getActivationPrime()(Z.back()).array() * (A.back() - _Y[i]).array();
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