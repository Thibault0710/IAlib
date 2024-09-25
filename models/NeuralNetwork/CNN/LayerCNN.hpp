#pragma once

#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include <execution>
#include "../Layer1D.hpp"
#include "../../../utils/vectorOperations.hpp"
#include "../../../utils/logistic.hpp"
#include "../../../utils/activation.hpp"
using Eigen::Matrix;
using namespace std;

template<typename U, typename V, typename W>
struct triplet {
    U first;
    V second;
    W third;
};

constexpr int MODE_ZERO_PADDING  = 0; // On ajoute des zeros sur les bords
constexpr int MODE_VALID_PADDING = 1; // On ne met pas de padding la taille de la matrice diminue

template<typename T>
class LayerCNN {
protected : 
    triplet<size_t, size_t, size_t> _input_dim;

public :
    triplet<size_t, size_t, size_t> getInputDim() const {
        return this->_input_dim;
    }

    virtual vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> feedforward(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input) const;

/*    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> backpropagation(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Z, const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &delta) const {
        auto Z_vec     = to_vector(Z);
        auto delta_vec = to_vector(delta);
        Matrix<T, Eigen::Dynamic, 1> ret = this->_lay.backpropagation(Z_vec, delta_vec);
        return to_matrix(ret, _input_dim.first, _input_dim.second, _input_dim.third);
    }

    void update(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &delta, const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input, T learning_rate) {
        auto delta_vec = to_vector(delta);
        auto input_vec = to_vector(input);
        this->_lay.update(delta_vec, input_vec, learning_rate);
    }

    void print() const {
        auto weight = this->getWeights();
        auto bias   = this->getBias();
        cout << "Weights : " << endl;
        for(size_t i = 0; i < weight.size(); ++i) cout << weight[i] << endl;
        cout << "Bias : " << endl;
        for(size_t i = 0; i < bias.size(); ++i) cout << bias[i] << endl;
    }*/
};