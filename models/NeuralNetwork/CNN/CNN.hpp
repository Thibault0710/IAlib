#pragma once

#include "LayerConv.hpp"
#include "Layer3D.hpp"
#include <execution>
#include <variant>
using Eigen::Matrix;
using namespace std;

template<typename T>
class CNN {
private: 
    triplet<size_t, size_t, size_t> _input_dim;
    triplet<size_t, size_t, size_t> _output_dim;
    vector<variant<LayerConv<T>, Layer3D<T>>> _net;
    vector< vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> >_X;
    vector< vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> >_Y;

public:
    CNN() : _input_dim({0,0,0}), _output_dim({0,0,0}) {}

    CNN(const triplet<size_t, size_t, size_t> &input_dim, const triplet<size_t, size_t, size_t> &output_dim) : _input_dim(input_dim), _output_dim(output_dim) {}

    CNN(const vector<variant<LayerConv<T>, Layer3D<T>>> &net) : _net(net) {
        for(size_t i = 1; i < _net.size(); ++i) {
        auto inputDim = visit([](auto &lay){ return lay.getInputDim(); }, _net[i]);
        auto outputDim = visit([](auto &lay){ return lay.getOutputDim(); }, _net[i-1]);
        
        if(inputDim.first != outputDim.first)
            throw invalid_argument("CNN::CNN(net) bad input sizes in first dim");
        if(inputDim.second != outputDim.second)
            throw invalid_argument("CNN::CNN(net) bad input sizes in second dim");
        if(inputDim.third != outputDim.third)
            throw invalid_argument("CNN::CNN(net) bad input sizes in third dim");
        }
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
        for(size_t i = 0; i < _net.size(); ++i) Y = visit([Y](auto &l){ return l.feedforward(Y); }, _net[i]);
        return Y;
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

    vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> feedforward(const vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> &X) {
        vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> Y(X.size());
        for(size_t i = 0; i < X.size(); ++i) Y[i] = this->feedforward(X[i]);
        return Y;
    }

    void fit(size_t epochs = 10, T learning_rate = 0.01) {
        for(size_t epoch = 0; epoch < epochs; ++epoch) {
            shuffleRows(_X, _Y);

            for(size_t i = 0; i < _X.size(); ++i) {
                vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> Z(this->_net.size() + 1);
                vector<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> A(this->_net.size() + 1);
                A[0] = _X[i];
                Z[0] = _X[i];

                for(size_t j = 0; j < this->_net.size(); ++j) {
                    Z[j + 1]     = visit([&A, &j](auto& ly){ return ly.feedforwardNoApply(A[j]); }, _net[j]);
                    A[j + 1]     = visit([&j, &Z](auto& ly){ return ly.getActivation()(Z[j+1]); }, _net[j]);
                }

                std::vector<std::vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> delta(this->_net.size()+1);
                delta.back() = A.back();
                auto deriv   = visit([&Z](auto &ly){ return ly.getActivationPrime()(Z.back()); }, _net.back());

                for(size_t k = 0; k < delta.back().size(); ++k) 
                    delta.back()[k] = deriv[k].array() * (A.back()[k] - _Y[i][k]).array();
                for(size_t j = 1; j < this->_net.size()+1; ++j)
                    delta[delta.size() - 1 - j] = visit([&Z, &delta, &A, &j](auto &ly){ return ly.backpropCNN(Z[Z.size()-1-j], delta[delta.size() - j], A[Z.size()-1-j]); }, _net[_net.size() - j]);

                for(size_t j = 0; j < this->_net.size(); ++j) 
                    visit([&delta, &A, &learning_rate, &j, &Z](auto &ly) {ly.updateCNN(delta[j+1], A[j], Z[j], learning_rate); }, _net[j]);
            }

            T total_error = error(_X, _Y);
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Error: " << total_error << std::endl;
        }
    }

    void print() const {
        for(size_t i =0; i < _net.size(); ++i) {
            cout << "Layer "<< i << ": " << endl;
            visit([](auto &ly){ ly.print(); }, _net[i]);
        }
    }

};