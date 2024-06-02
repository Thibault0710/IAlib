#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "./utils/logistic.hpp"
using Eigen::Matrix;

template<typename T>
class KMeans {
private :
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _X; // The Data
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _centroids; // The centroids of the model
    size_t _K; // The number of cluster we aim to have
    const size_t maxIteration  = 100000;
    const double stopCondition = 1e-3;

    void initialiseCentroids() {
        if(_X.rows() < _K) std::cerr << "You do not have enought data to make the prediction precise" << std::endl;
        for(size_t i = 0; i < _K; ++i) _centroids.row(i) = _X.row(i);
    }

    size_t findCluster(const Matrix<T, Eigen::Dynamic, 1> &data) const {
        double smallestDistance = std::numeric_limits<double>::infinity();
        size_t cluster          = 0;

        for(size_t k = 0; k < _K; ++k) {
            auto distanceCentroid = (data - ((Matrix<T, Eigen::Dynamic, 1>) _centroids.row(k))).norm();
            if(distanceCentroid < smallestDistance) {
                smallestDistance = distanceCentroid;
                cluster          = k;
            }
        }
        return cluster;
    }

    bool updateCentroids() {
        // TODO : change the return true ti find a more satisfying stop condition than the number of iteration
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> pastCentroids = _centroids;
        std::vector<Matrix<T, Eigen::Dynamic, 1>> indexInverse(_K, Matrix<T, Eigen::Dynamic, 1>::Zero(_X.cols()));
        std::vector<size_t> elementsParCluster(_K, 0);


        for (size_t i = 0; i < _X.rows(); ++i) {
            size_t cluster         = this->findCluster(_X.row(i));
            indexInverse[cluster] += _X.row(i).transpose();
            ++elementsParCluster[cluster];
        }

        for (size_t i = 0; i < _K; ++i) {
            if (elementsParCluster[i] > 0) _centroids.row(i) = indexInverse[i].transpose() / elementsParCluster[i];
            else _centroids.row(i).setZero();
        }
        return (pastCentroids - _centroids).norm() > stopCondition;
    }


public :
    KMeans(const Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, size_t K) : _X(X), _K(K) {
            _centroids.resize(_K, X.cols());
    }

    KMeans(const std::vector<std::vector<T>> &X, size_t K) {
            size_t rows = X.size();
            size_t cols = X[0].size();

            _X.resize(rows, cols);
            for(size_t i = 0; i < rows; ++i) for(size_t j = 0; j < cols; ++j) _X(i, j) = X[i][j];
            _K = K;
            _centroids.resize(_K, cols);

    }

    void fit() { // This method compute the centroids of the model
        initialiseCentroids();

        bool pass = true;
        for(size_t i = 0; i < maxIteration && pass; ++i) pass = updateCentroids();
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getData() const {
        return _X;
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getCentroids() const {
        return _centroids;
    }
};