#pragma once
#include "Layer3D.hpp"

using Eigen::Matrix;

template<typename T>
class NeuralNetwork3D {
private : 
    triplet<size_t, size_t, size_t> _input_dim;
    triplet<size_t, size_t, size_t> _output_dim;
    std::vector<Layer3D<T>> _net;
    vector< vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> > _X;
    vector< vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> > _Y;

public : 
    NeuralNetwork3D(const std::vector<triplet<size_t, size_t, size_t>> &tailles) : _input_dim(tailles.front()), _output_dim(tailles.back()), _net(tailles.size() - 1) {
        for(size_t i = 0; i < tailles.size() - 1; ++i) _net[i] = Layer3D<T>(tailles[i], tailles[i+1]); // On garde la sigmoid pour l'instant et on pourra faire d'autres consturecteurs apres
    }

    NeuralNetwork3D(const vector<Layer3D<T>> &net) : _input_dim(net[0].getInputDim()), _output_dim(net.back().getOutputDim()), _net(net) {}

    std::vector<std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> getWeights() const {
        std::vector<std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> weights(_net.size());
        for(size_t i = 0; i < _net.size(); ++i) weights[i] = this->_net[i].getWeights();
        return weights;
    }

    void loadData(const vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> &X, const vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> &Y) {
        _X = X;
        _Y = Y;
    }

    void clearData() {
        _X.clear();
        _Y.clear();
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> feedforward(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &X) {
        vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> Y(X);
        for(size_t i = 0; i < _net.size(); ++i) Y = _net[i].feedforward(Y);
        return Y;
    }

    vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> feedforward(const vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> &X) {
        vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> Y(X.size());
        for(size_t i = 0; i < X.size(); ++i) Y[i] = this->feedforward(X[i]);
        return Y;
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> feedforwardPrint(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &X) {
        vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> Y(X);
        for(size_t i = 0; i < _net.size(); ++i) {
            cout << "Layer: " << i << endl;
            for(auto &y : Y) cout << y << endl;
            Y = _net[i].feedforward(Y);
        }
        cout << "Last layer: " << endl;
        for(auto &y : Y) cout << y << endl;
        return Y;
    }

    double score(const vector< vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> >& X, const vector< vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> >& Y) {
        double score = 0.0;
        for(size_t i = 0; i < X.size(); ++i) {
            vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> prediction = this->feedforward(X[i]);
            
            int maxIndexPred = 0, maxIndexLabel = 0;
            T maxValPred = prediction[0](0, 0), maxValLabel = Y[i][0](0, 0);
            for(int index = 0; index < prediction.size(); ++index) {
                for (int row = 0; row < prediction[index].rows(); ++row) {
                    for (int col = 0; col < prediction[index].cols(); ++col) {
                        if (prediction[index](row, col) > maxValPred) {
                            maxValPred   = prediction[index](row, col);
                            maxIndexPred = row * prediction[index].cols() + col;
                        }
                        if (Y[i][index](row, col) > maxValLabel) {
                            maxValLabel   = Y[i][index](row, col);
                            maxIndexLabel = row * Y[i][index].cols() + col;
                        }
                    }
                }
            }
            score += (maxIndexLabel == maxIndexPred) ? 1.0 : 0.0;
        }
        return score / X.size();
    }

    T error(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &X, const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Y) {
        T ret = 0;
        auto predicted = this->feedforward(X);
        for(size_t j = 0; j < predicted.size(); ++j) ret += (predicted[j] - Y[j]).squaredNorm();
        return ret;
    }

    T error(const vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> &X, const vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> &Y) {
        T ret = 0;
        for(size_t i = 0; i < X.size(); ++i) {
            ret += this->error(X[i], Y[i]);
        }
        return ret / static_cast<T>(X.size());
    }

    void fit(size_t epochs = 10, T learning_rate = 0.01) {
        for(size_t epoch = 0; epoch < epochs; ++epoch) {
            shuffleRows(_X, _Y);

            for(size_t i = 0; i < _X.size(); ++i) {
                // Passe avant
                vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> Z(this->_net.size() + 1);
                vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> A(this->_net.size() + 1);
                A[0] = _X[i];
                Z[0] = _X[i];

                for(size_t j = 0; j < this->_net.size(); ++j) {
                    Z[j+1] = this->_net[j].feedforwardNoApply(A[j]);
                    A[j+1] = _net[j].getActivation()(Z[j+1]);
                }

                vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> delta(this->_net.size() + 1);
                delta.back() = A.back();
                auto deriv =  _net.back().getActivationPrime()(Z.back());
                
                for(size_t k = 0; k < delta.back().size(); ++k) 
                    delta.back()[k] = deriv[k].array() * (A.back()[k] - _Y[i][k]).array();
                for(size_t j = 1; j < this->_net.size()+1; ++j) 
                    delta[delta.size() - 1 - j] = _net[_net.size() - j].backpropagation(Z[Z.size()-1-j], delta[delta.size() - j]);
                for(size_t j = 0; j < this->_net.size(); ++j) _net[j].update(delta[j+1], A[j], learning_rate);
            }

            T total_error = error(_X, _Y);
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Error: " << total_error << std::endl;
        }
    }
    
    void print() {
        for(size_t i = 0; i < _net.size(); ++i) {
            std::cout << "Layer : "<< i << endl;
            _net[i].print();
        }
    }

};