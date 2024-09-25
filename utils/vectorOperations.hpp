#ifndef VECTOR_OPERATIONS_HPP
#define VECTOR_OPERATIONS_HPP

#include <eigen3/Eigen/Dense>
#include <bits/stdc++.h>
#include "csv.hpp"

template<typename U, typename V, typename W>
struct triplet {
    U first;
    V second;
    W third;
};

template<typename T>
void printt(const std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &vec) {
    for(size_t i = 0; i < vec.size(); ++i) {
        std::cout << "Couche: " << i<<std::endl<<vec[i] << std::endl;
    }
}

bool matricesAreEqual(const Eigen::MatrixXd& eigenMatrix, const std::vector<std::vector<double>>& vectorOfVectors) {
    if(eigenMatrix.rows() != (long int) vectorOfVectors.size() || eigenMatrix.cols() != (long int) vectorOfVectors[0].size()) return false;
    for(int i = 0; i < eigenMatrix.rows(); ++i) for(int j = 0; j < eigenMatrix.cols(); ++j) if(eigenMatrix(i, j) != vectorOfVectors[i][j]) return false;

    return true;
}

template <typename T>
std::vector<std::vector<T>> toOneVector(const std::vector<T> &vec) {
        std::vector<std::vector<T>> ret(vec.size(), std::vector<T>(1));
        for(size_t i = 0; i < vec.size(); ++i) ret[i][0] = vec[i];
        return ret;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> vectorToEigenMatrix(const std::vector<std::vector<T>>& vec) {
    if (vec.empty()) return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>();

    size_t rows = vec.size();
    size_t cols = vec[0].size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat(rows, cols);

    for (size_t i = 0; i < rows; ++i) for(size_t j = 0; j < cols; ++j) mat(i, j) = vec[i][j];
    return mat;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> vectorToEigenMatrix(const std::vector<T>& vec) {
    if (vec.empty()) return Eigen::Matrix<T, Eigen::Dynamic, 1>();

    size_t rows = vec.size();
    Eigen::Matrix<T, Eigen::Dynamic, 1> mat(rows, 1);

    for (size_t i = 0; i < rows; ++i) mat(i, 0) = vec[i];
    return mat;
}

template<typename T>
std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> vectorToEigenMatrix(const std::vector<std::vector<std::vector<T>>>& vec) {
    if (vec.empty()) return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>();

    size_t data_size = vec.size();
    size_t rows      = vec[0].size();
    size_t cols      = vec[0][0].size();
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> mat(data_size, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(rows, cols));

    for (size_t i = 0; i < data_size; ++i) for(size_t j = 0; j < rows; ++j) for(size_t k = 0; k < cols; ++k) mat[i](j, k) = vec[i][j][k];
    return mat;
}

template<typename T>
T mostFrequentValue(const std::vector<T> &vec) {
    if (vec.empty()) throw std::invalid_argument("vectorOperations.hpp::mostFrequentValue : Vector is empty.");
    std::unordered_map<T, int> mp;
    for(size_t i = 0; i < vec.size(); ++i) ++mp[vec[i]];

    return std::max_element(mp.begin(), mp.end(), [](const auto &a, const auto &b) {return a.second < b.second;} )->first;
}

template<typename T>
void shuffleRows(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &Y) {
    std::vector<size_t> indices(X.rows());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> X_shuffle(X.rows(), X.cols());
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Y_shuffle(Y.rows(), Y.cols());

    for(size_t i = 0; i < X.rows(); ++i) {
        X_shuffle.row(i) = X.row(indices[i]);
        Y_shuffle.row(i) = Y.row(indices[i]);
    }

    X = X_shuffle;
    Y = Y_shuffle;
}

template<typename T>
void shuffleRows(std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &X, std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &Y) {
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> X_shuffle(X.size(), Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(X[0].rows(), X[0].cols()));
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> Y_shuffle(X.size(), Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(Y[0].rows(), Y[0].cols()));

    for(size_t i = 0; i < X.size(); ++i) {
        X_shuffle[i] = X[indices[i]];
        Y_shuffle[i] = Y[indices[i]];
    }

    X = X_shuffle;
    Y = Y_shuffle;
}

template<typename T>
void shuffleRows(std::vector<std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> &X, std::vector<std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> &Y) {
    std::vector<size_t> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> X_shuffle(X.size());
    std::vector<std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>> Y_shuffle(Y.size());
        
    for(size_t i = 0; i < X.size(); ++i) {
        X_shuffle[i] = X[indices[i]];
        Y_shuffle[i] = Y[indices[i]];
    }

    X = std::move(X_shuffle);
    Y = std::move(Y_shuffle);
}

std::vector<std::vector<double>> one_hot_encode(const std::vector<double>& labels, size_t num_classes = 10) {
    std::vector<std::vector<double>> encoded_labels(labels.size(), std::vector<double>(num_classes, 0.0));
    
    for (size_t i = 0; i < labels.size(); ++i) {
        size_t label_index = static_cast<size_t>(labels[i]);
        encoded_labels[i][label_index] = 1.0;
    }
    
    return encoded_labels;
}

std::vector<double> one_hot_encode(double label, size_t num_classes = 10) {
    std::vector<double> encoded_labels(num_classes, 0.0);
    size_t label_index          = static_cast<size_t>(label);
    encoded_labels[label_index] = 1.0;
    return encoded_labels;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> prepareData(const std::string &filename) {
    std::vector<std::vector<double>> train = read_csv(filename);
    std::vector<std::vector<double>> labels(train.size());
    std::vector<std::vector<double>>   data(train.size());

    std::transform(train.begin(), train.end(), labels.begin(), [](const std::vector<double> &line){ return one_hot_encode(line[0]); });
    std::transform(train.begin(), train.end(),   data.begin(), [](const std::vector<double>& row) {
                       std::vector<double> temp(row.begin() + 1, row.end());
                       return temp;
                   });

    return {data, labels};
}

std::pair<std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>, std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> prepareData2D(const std::string &filename, size_t i1, size_t i2, size_t o1, size_t o2) {
    auto [data, labels] = prepareData(filename);
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> datas(data.size(), Eigen::Matrix<double, -1,-1>(i1,i2));
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> lab(data.size(), Eigen::Matrix<double, -1,-1>(o1,o2));
    for(size_t k = 0; k < data.size(); ++k){
        for(size_t i = 0; i < i1; ++i) for(size_t j = 0; j < i2; ++j) datas[k](i,j) = data[k][i*i2+j]/255.0;
        for(size_t i = 0; i < o1; ++i) for(size_t j = 0; j < o2; ++j) lab[k](i,0) = labels[k][i*o2+j];
    }
    return {datas, lab};
}

std::pair<std::vector<std::vector<double>>, std::vector<std::string>> prepareDataIris(const std::string &filename) {
    std::vector<std::vector<std::string>> file = read_csv_string(filename);
    std::vector<std::vector<double>> data(file.size());
    std::vector<std::string> labels(file.size());

    std::transform(file.begin(), file.end(), data.begin(), [](std::vector<std::string> vec){std::vector<double> ret(vec.size() - 1);
                                                                                            std::transform(vec.begin(), vec.end()-1, ret.begin(), [](std::string str){ return std::stod(str); });
                                                                                            return ret;
                                                                                            });
    std::transform(file.begin(), file.end(), labels.begin(), [](std::vector<std::string> vec){ return vec.back(); });
    return {data, labels};
}

template<typename T>
std::pair<std::vector<std::vector<std::vector<T>>>, std::vector<std::vector<int>>> bootstrapSamples(const std::vector<std::vector<T>>& data, const std::vector<int> &labels, size_t num_trees, size_t num_samples) {
    std::vector<std::vector<std::vector<T>>> X_trees(num_trees, std::vector<std::vector<T>>(num_samples));
    std::vector<std::vector<int>> Y_trees(num_trees, std::vector<int>(num_samples));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    for(size_t i = 0; i < num_trees; ++i) {
        for(size_t j = 0; j < num_samples; ++j) {
            auto index    = dis(gen);
            X_trees[i][j] = data[index];
            Y_trees[i][j] = labels[index];
        }
    }

    return {X_trees, Y_trees};
}

template<typename T>
std::vector<std::vector<T>> normalize(const std::vector<std::vector<T>> &vec) {
    std::vector<std::vector<T>> copyy(vec.size());
    std::transform(vec.begin(), vec.end(), copyy.begin(), [](std::vector<T> v){ T summ = std::accumulate(v.begin(), v.end(), 0.0);
                                                                                std::transform(v.begin(), v.end(), [&summ](T x){ return x/summ; });
                                                                                return v;
                                                                            });
    return copyy;
}

template<typename U, typename V>
std::tuple<std::vector<std::vector<U>>, std::vector<V>, std::vector<std::vector<U>>, std::vector<V>> test_train_split(const std::vector<std::vector<U>> &data, const std::vector<V> &labels, double percent) {
    std::vector<int> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    size_t train_size = (size_t) (percent * ((double) data.size()));
    size_t test_size  = data.size() - train_size;
    std::vector<std::vector<U>> data_train(train_size);
    std::vector<std::vector<U>> data_test (test_size);
    std::vector<V> labels_train(train_size);
    std::vector<V> labels_test(test_size);

    for(size_t i = 0; i < train_size; ++i) {
        data_train[i]   = data[indices[i]];
        labels_train[i] = labels[indices[i]];
    }
    for(size_t i = train_size; i < data.size(); ++i) {
        data_test[i - train_size]   = data[indices[i]];
        labels_test[i - train_size] = labels[indices[i]];
    }

    return {data_train, labels_train, data_test, labels_test};
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> to_matrix(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vec, size_t d1, size_t d2) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> ret(d1, d2);
    for(size_t i = 0; i < d1; ++i) for(size_t j = 0; j < d2; ++j) ret(i, j) = vec(i*d2 + j, 0);
    return ret;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> to_vector(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &mat) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> ret(mat.rows() * mat.cols());
    for(size_t i = 0; i < mat.rows(); ++i) for(size_t j = 0; j < mat.cols(); ++j) ret(i*mat.cols() + j, 0) = mat(i, j);
    return ret;
}

template<typename T>
std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> to_matrix(const Eigen::Matrix<T, Eigen::Dynamic, 1> &vec, size_t d1, size_t d2, size_t d3) {
    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ret(d1, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(d2, d3));
    for(size_t i = 0; i < d1; ++i) 
        for(size_t j = 0; j < d2; ++j) 
            for(size_t k = 0; k < d3; ++k)
                ret[i](j, k) = vec((i*d2 + j)*d3 + k, 0);
    return ret;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> to_vector(const std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &mat) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> ret(mat[0].rows() * mat[0].cols()*mat.size());
    for(size_t i = 0; i < mat.size(); ++i) 
        for(size_t j = 0; j < mat[0].rows(); ++j)
            for(size_t k = 0; k < mat[0].cols(); ++k)
                ret((i*mat[0].rows() + j)*mat[0].cols() + k, 0) = mat[i](j, k);
    return ret;
}

template<typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> modify_padding(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &X, int nbe_zero_pad) { // nbe_zero_pad corresponds to the number of the we add on the border of the picture
    if(nbe_zero_pad == 0) return X;
    
    int rows = X.rows();
    int cols = X.cols();
    int new_rows = rows + 2 * nbe_zero_pad;
    int new_cols = cols + 2 * nbe_zero_pad;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> padded_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(new_rows, new_cols);
    padded_matrix.block(nbe_zero_pad, nbe_zero_pad, rows, cols) = X;
    
    return padded_matrix;
}

#endif 