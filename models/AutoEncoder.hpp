#include "NeuralNetwork/NeuralNetwork.hpp"
using Eigen::Dynamic;

template<typename T>
class AutoEncoder{
private:
    size_t _input_dim;
    size_t _projection_dim;
    NeuralNetwork<T> _net;
    Matrix<T, Dynamic, Dynamic> _data;

public:
    AutoEncoder() : _input_dim(0), _projection_dim(0) {}

    AutoEncoder(size_t input_dim, size_t projection_dim) : _input_dim(input_dim), 
                                                           _projection_dim(projection_dim),
                                                           _net({Layer1D<T>(input_dim, projection_dim, "reLu"),
                                                                 Layer1D<T>(projection_dim, input_dim, "reLu")}) {}

    void loadData(const Matrix<T, Dynamic, Dynamic> &X) {
        _net.loadData(X, X);
    }

    void loadData(const vector<vector<T>> &X) {
        _net.loadData(X, X);
    }

    void fit(size_t epochs=10, T learning_rate=0.01) {
        _net.fit(epochs, learning_rate);
    }

    Matrix<T, Dynamic, 1> feedforward(const Matrix<T, Dynamic, Dynamic> &input) {
        return this->_net.feedforwardGetWeights(input, 1);
    }

    Matrix<T, Dynamic, 1> feedforward(const vector<T> &input) {
        return this->_net.feedforwardGetWeights(input, 1);
    }

};