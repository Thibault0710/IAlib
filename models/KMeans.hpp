#include <bits/stdc++.h>
#include <eigen3/Eigen/Dense>
#include "../utils/logistic.hpp"
#include "../utils/activation.hpp"
using Eigen::Matrix;

template<typename T>
class KMeans {
private :
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _X; // The Data
    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> _centroids; // The centroids of the model
    size_t _K; // The number of cluster we aim to have
    const size_t maxIteration  = 100000;
    const double stopCondition = 1e-3;

    void initialiseCentroids(double mean = 0.0, double stddev = -1.0) {
        if(_X.rows() < _K) std::cerr << "You do not have enought data to make the prediction precise" << std::endl;

        if(stddev == -1) {
            for(size_t i = 0; i < _K; ++i) _centroids.row(i) = _X.row(i);
            return ;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> d(mean, stddev);

        _centroids.resize(_K, _X.cols());
        for (size_t i = 0; i < _K; ++i) for (size_t j = 0; j < _X.cols(); ++j) _centroids(i, j) = d(gen);
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

    void fit(double mean = 0.0, double stddev = -1.0) { // This method compute the centroids of the model
        initialiseCentroids(mean, stddev);

        bool pass = true;
        for(size_t i = 0; i < maxIteration && pass; ++i) pass = updateCentroids();
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getData() const {
        return _X;
    }

    Matrix<T, Eigen::Dynamic, Eigen::Dynamic> getCentroids() const {
        return _centroids;
    }

    Matrix<size_t, Eigen::Dynamic, 1> getClasses() const { // returns the classes associated with clusters for every data (i.e a line in _X) classes are in [0, _K[
        Matrix<size_t, Eigen::Dynamic, 1> classes(_X.rows());
        for(size_t i = 0; i < _X.rows(); ++i) classes(i, 0) = this->findCluster(_X.row(i));
        return classes;
    }    

    std::vector<std::vector<size_t>> getIndexInverted() const { // returns a vector of length _K at each index corresponding to a cluster in associate the vector of data associated to this cluster
        std::vector<std::vector<size_t>> index(_K);
        auto classes = this->getClasses();
        for(size_t i = 0; i < classes.rows(); ++i) index[classes[i]].push_back(i);
        return index;
    }

    Matrix<double, Eigen::Dynamic, 1> variance() const {// Computes the variance intra-cluster
        Matrix<double, Eigen::Dynamic, 1> variances(_K);
        auto index = this->getIndexInverted();
        for(size_t k = 0; k < _K; ++k) {
            for(size_t i = 0; i < index[k].size(); ++i) {
                variances(k, 0) += (_X.row(index[k][i]) - _centroids.row(k)).squaredNorm();
            }
            if(index[k].size() > 0) variances(k, 0) /= index[k].size();
        }
        return variances;
    }

    double inertie() const {// Computes the variance intra-cluster
        double inertie = 0.0;
        auto index     = this->getIndexInverted();
        for(size_t k = 0; k < _K; ++k) for(size_t i = 0; i < index[k].size(); ++i) inertie += (_X.row(index[k][i]) - _centroids.row(k)).squaredNorm();
        return inertie;
    }

    Matrix<double, Eigen::Dynamic, 1> elbow(size_t Kmax = 20) { // returns the value of inertie for different K value, line i contains the inertie for _K = i
        size_t stop = std::min(Kmax, (size_t) _X.rows());
        size_t tmp = _K;
        Matrix<double, Eigen::Dynamic, 1> ret(stop);
        for(size_t k = 2; k < stop; ++k) {
            this->_K = k;
            this->fit();
            ret[k-2]   = this->inertie(); // We compute the inertie into each class
        }
        _K = tmp;
        return ret;
    }

};