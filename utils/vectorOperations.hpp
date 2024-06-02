#include <eigen3/Eigen/Dense>
#include <vector>

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
    if (vec.empty()) {
        return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>();
    }

    size_t rows = vec.size();
    size_t cols = vec[0].size();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = vec[i][j];
        }
    }

    return mat;
}

template<typename T>
T mostFrequentValue(const std::vector<T> &vec) {
    if (vec.empty()) throw std::invalid_argument("vectorOperations.hpp::mostFrequentValue : Vector is empty.");
    std::unordered_map<T, int> mp;
    for(size_t i = 0; i < vec.size(); ++i) ++mp[vec[i]];

    return std::max_element(mp.begin(), mp.end(), [](const auto &a, const auto &b) {return a.second < b.second;} )->first;
}