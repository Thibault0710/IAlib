#include <gtest/gtest.h>
#include "../models/NeuralNetwork/NeuralNetwork2D.hpp"
#include "../utils/csv.hpp"
#include "../utils/activation.hpp"
using namespace std;

TEST(NeuralNetwork2DTest, constructorTest) {
    vector<pair<size_t, size_t>> size {{5,5}, {3,3}, {6,6}};
    NeuralNetwork2D<double> nn(size);
}

TEST(NeuralNetwork2DTest, feedforwardTest) {
    vector<pair<size_t, size_t>> size {{5,5}, {3,3}, {6,6}};
    NeuralNetwork2D<double> nn(size);
    Matrix<double, 5, 5> X;
    X << 5, 2, 5, 2, 5,
         5, 5, 0, 2, 6,
         8, 9, 8, 0, 0,
         0, 1, 5, 2, 6,
         7, 4, 6, 5, 4;
    auto Y = nn.feedforward(X);
    ASSERT_EQ(Y.rows(), 6);
    ASSERT_EQ(Y.cols(), 6);
}

TEST(NeuralNetwork2DTest, errorTest) {
    vector<pair<size_t, size_t>> size {{5,5}, {3,3}, {6,6}};
    NeuralNetwork2D<double> nn(size);
    Matrix<double, 5, 5> X;
    X << 5, 2, 5, 2, 5,
         5, 5, 0, 2, 6,
         8, 9, 8, 0, 0,
         0, 1, 5, 2, 6,
         7, 4, 6, 5, 4;
    Matrix<double, 6, 6> Y;
    Y << 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1;
    auto error = nn.error(X,Y);
}

TEST(NeuralNetwork2DTest, errorTest2) {
    vector<pair<size_t, size_t>> size {{5,5}, {3,3}, {6,6}};
    NeuralNetwork2D<double> nn(size);
    Matrix<double, 5, 5> X;
    X << 5, 2, 5, 2, 5,
         5, 5, 0, 2, 6,
         8, 9, 8, 0, 0,
         0, 1, 5, 2, 6,
         7, 4, 6, 5, 4;
    Matrix<double, 6, 6> Y;
    Y << 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1;
    vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> inputs {X, X};
    vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> outputs {Y, Y};
    auto error = nn.error(inputs, outputs);
}

TEST(NeuralNetwork2DTest, loadDataTest) {
    vector<pair<size_t, size_t>> size {{5,5}, {3,3}, {6,6}};
    NeuralNetwork2D<double> nn(size);
    Matrix<double, 5, 5> X;
    X << 5, 2, 5, 2, 5,
         5, 5, 0, 2, 6,
         8, 9, 8, 0, 0,
         0, 1, 5, 2, 6,
         7, 4, 6, 5, 4;
    Matrix<double, 6, 6> Y;
    Y << 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1;
    vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> inputs {X, X};
    vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> outputs {Y, Y};
    nn.loadData(inputs, outputs);
}

TEST(NeuralNetwork2DTest, fitTest) {
    vector<pair<size_t, size_t>> size {{5,5}, {3,3}, {6,6}};
    NeuralNetwork2D<double> nn(size);
    Matrix<double, 5, 5> X;
    X << 5, 2, 5, 2, 5,
         5, 5, 0, 2, 6,
         8, 9, 8, 0, 0,
         0, 1, 5, 2, 6,
         7, 4, 6, 5, 4;
    Matrix<double, 6, 6> Y;
    Y << 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1;
    vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> inputs {X, X};
    vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> outputs {Y, Y};
    nn.loadData(inputs, outputs);
    nn.fit();
}

TEST(NeuralNetwork2DTest, mnistTest) {
    string train {"MNIST/mnist_train.csv"};
    vector<pair<size_t, size_t>> sizes {{28, 28},{128, 1},{10, 1}};
    NeuralNetwork2D<double> nn(sizes);

    auto [datas, lab] = prepareData2D(train, 28, 28, 10, 1);
    nn.loadData(datas, lab);
    nn.fit();
    nn.feedforwardPrint(datas[1]);
    
    string test {"MNIST/mnist_test.csv"};
    auto [X, Y] = prepareData2D(test, 28, 28, 10, 1);
    double score = nn.score(X, Y);
    cout <<endl<<endl<< score <<endl;
    ASSERT_GE(score, 0.60);
}

TEST(NeuralNetwork2DTest, mnistTestMultiLayers) {
    string train {"MNIST/mnist_train.csv"};
    auto [data, labels]     = prepareData2D(train, 28, 28, 10, 1);

    vector<Layer2D<double>> net  {  Layer2D<double>({28, 28},  {128, 1}, "reLu"),
                                    Layer2D<double>({128, 1},  {10, 1}, "reLu")};
    NeuralNetwork2D<double> nn(net);

    nn.loadData(data, labels);
    nn.fit();

    string test {"MNIST/mnist_test.csv"};
    auto pr2 = prepareData2D(test, 28, 28, 10, 1);
    double score = nn.score(pr2.first, pr2.second);
    cout << "score :" <<score<<endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}