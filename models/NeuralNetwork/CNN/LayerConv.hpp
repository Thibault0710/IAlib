#pragma once

#include <Eigen/Dense>
#include <bits/stdc++.h>
#include <execution> 
#include "../../../utils/vectorOperations.hpp"
#include "../../../utils/activation.hpp"
using Eigen::Matrix;
using namespace std;

constexpr int MODE_ZERO_PADDING  = 0; // On ajoute des zeros sur les bords
constexpr int MODE_VALID_PADDING = 1; // On ne met pas de padding la taille de la matrice diminue

template<typename T>
class LayerConv{
private:
    pair<size_t, size_t> _input_filter_dim;
    triplet<size_t, size_t, size_t> _input_dim;
    size_t _nbe_filters;
    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> _filters;

public:
    LayerConv() : _nbe_filters(0), _input_dim({0, 0, 0}) {}

    LayerConv(const pair<size_t, size_t> &input_filter_dim, const triplet<size_t, size_t, size_t> &input_dim, size_t nbe_filters = 64)
        : _input_filter_dim(input_filter_dim), _input_dim(input_dim), _nbe_filters(nbe_filters), _filters(nbe_filters) {
        
        // Distribution normale de Glorot
        size_t fan_in = input_filter_dim.first * input_filter_dim.second * input_dim.third;
        size_t fan_out = _nbe_filters * input_filter_dim.first * input_filter_dim.second;
        T stddev = std::sqrt(2.0 / (fan_in + fan_out));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(0.0, stddev);

        // Initialisation des filtres avec la distribution normale de Glorot
        for (auto& filter : _filters) {
            filter = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::NullaryExpr(
                _input_filter_dim.first, _input_filter_dim.second, [&]() { return dist(gen); });
        }
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> applyFilter(Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X, size_t filterIndex) const {
        auto l1 = _input_dim.second - _input_filter_dim.first + 1;
        auto l2 = _input_dim.third - _input_filter_dim.second + 1;
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> result(l1, l2);
        for(size_t i = 0; i < l1; ++i) 
            for(size_t j = 0; j < l2; ++j) 
                result(i, j) = (X.block(i, j, _input_filter_dim.first, _input_filter_dim.second).array() * _filters[filterIndex].array()).sum();
        return result;
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getFilter(size_t index_filter) const {
        if(index_filter >= _filters.size()) throw std::invalid_argument("LayerConv::getFilter(index_filter) : index_filter is too large compared to the number of filters");
        return _filters[index_filter];
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> getFilters() const {
        return this->_filters;
    }


    function<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>)> getActivation() const {
        return get3DFunction<T>("identite");
    }
    
    function<vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>)> getActivationPrime() const {
        return get3DFunction<T>("identitePrime");
    }

    triplet<size_t, size_t, size_t> getInputDim() const {
        return this->_input_dim;
    }

    triplet<size_t, size_t, size_t> getOutputDim() const {
        return {_input_dim.first * _nbe_filters, _input_dim.second - _input_filter_dim.first + 1, _input_dim.third - _input_filter_dim.second + 1};
    }

    size_t getNbeFilters() const {
        return this->_nbe_filters;
    }

    void setFilterConstant(T value, size_t index_filter) {
        if(index_filter >= _filters.size()) throw std::invalid_argument("LayerConv::getFilter(index_filter) : index_filter is too large compared to the number of filters");
        this->_filters[index_filter].setConstant(value);
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> feedforward(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X) const { // Renvoie un tenseur
        vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ret(_nbe_filters);
        for(size_t i = 0; i < _nbe_filters; ++i) ret[i] = this->applyFilter(X, i);
        return ret;
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> feedforward(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &X) const {
        vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ret(_nbe_filters * X.size());
        for(size_t i = 0; i < X.size(); ++i) for(size_t j = 0; j < _nbe_filters; ++j) ret[i*_nbe_filters + j] = this->applyFilter(X[i], j);
        return ret;
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> feedforwardNoApply(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input) const {
        return feedforward(input);
    }

    void print() const {
        for(size_t i = 0; i < _nbe_filters; ++i) cout << "Filter " << i << ": " << endl << _filters[i] << endl;
    }

    vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> backpropCNN(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Z, 
                                                                    const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &delta,
                                                                    const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input) const {
        vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> past_input(_input_dim.first, Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(_input_dim.second, _input_dim.third));
        auto output_dim = this->getOutputDim();

        for(int q = 0; q < _input_dim.first; ++q) {
            past_input[q].setZero();
            for(int m = 0; m < _input_dim.second; ++m) {
                for(int n = 0; n < _input_dim.third; ++n) {
                    for(int r = 0; r < _nbe_filters; ++r) {
                        for(int j = max(m-((int) _input_filter_dim.first)+1, 0); j <= min(m, (int) output_dim.second-1); ++j) {
                            for(int k = max(n-((int) _input_filter_dim.second)+1, 0); k <= min(n, (int) output_dim.third-1); ++k) {
                                past_input[q](m, n) += delta[q*_nbe_filters + r](j, k) * _filters[r](m-j, n-k);
                            }
                        }
                    }
                }
            }
        }
        return past_input;
    }

    void updateCNN(const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &delta,
                   const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &input,
                   const vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &A,
                   T learning_rate) {
        /*
        On aurait pu calculer de nouveaux parametres dans génériques tels que delta dans le cadre de la retropropagation pour n'avoir
        plus qu'à calculer _filters[i] - learning_rate*potentiel_parametre.
        Mais ne voulant pas ajouter une couche de complexité a la fonction fit on calcul ce parametre directement ici. 
        Le calcul est semblable a celui effectué dans backpropInput au vu du calcul symétrique de la convolution
        */
        vector<Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> filter_gradients(_nbe_filters, Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(_input_filter_dim.first, _input_filter_dim.second));
        auto output_dim = this->getOutputDim();

        for(int r = 0; r < _nbe_filters; ++r) {
            filter_gradients[r].setZero();
            for(int p1 = 0; p1 < _input_filter_dim.first; ++p1) {
                for(int p2 = 0; p2 < _input_filter_dim.second; ++p2) {
                    for(int q = 0; q < _input_dim.first; ++q) {
                        for(int j = 0; j < output_dim.second; ++j) {
                            for(int k = 0; k < output_dim.third; ++k) {
                                filter_gradients[r](p1, p2) += delta[q*_nbe_filters + r](j, k) * A[q](j+p1, k+p2);
                            }
                        }
                    }
                }
            }
        }
        // Mettre à jour les filtres avec les gradients calculés
        for (size_t l = 0; l < _nbe_filters; ++l) {
            _filters[l] -= learning_rate * filter_gradients[l];
        }
    }
};