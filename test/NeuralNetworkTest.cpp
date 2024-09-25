#include <gtest/gtest.h>
#include "../models/NeuralNetwork/NeuralNetwork.hpp"
#include "../utils/csv.hpp"
#include "../utils/activation.hpp"
using namespace std;

TEST(NeuralNetworkTest, constructorTest) {
    vector<size_t> size {5,3,6};
    NeuralNetwork<double> nn(size);
}

TEST(NeuralNetworkTest, feedforwardTest) {
    vector<size_t> size {5,3,6};
    NeuralNetwork<double> nn(size);
    Matrix<double, 5, 1> X;
    X << 5,2,5,2,5;
    auto Y = nn.feedforward(X);
    ASSERT_EQ(Y.rows(), 6);
}

TEST(NeuralNetworkTest, errorTest) {
    vector<size_t> size {5,3,6};
    NeuralNetwork<double> nn(size);
    Matrix<double, 5, 1> X;
    X << 5,2,5,2,5;
    Matrix<double, 6, 1> Y;
    Y << 1,1,1,1,1,1;
    auto error = nn.error(X.transpose(),Y.transpose());
}

TEST(NeuralNetworkTest, errorTest2) {
    vector<size_t> size {5,3,6};
    NeuralNetwork<double> nn(size);
    Matrix<double, 5, 2> X;
    X << 5,2,5,2,5,  5,2,5,2,1;
    Matrix<double, 6, 2> Y;
    Y << 1,1,1,1,1,1, 1,1,1,1,1,1;
    auto er = nn.error(X.transpose(),Y.transpose());
}

TEST(NeuralNetworkTest, loadDataTest) {
    vector<size_t> size {5,3,6};
    NeuralNetwork<double> nn(size);
    Matrix<double, 5, 1> X;
    X << 5,2,5,2,5;
    Matrix<double, 6, 1> Y;
    Y << 1,1,1,1,1,1;
    nn.loadData(X.transpose(), Y.transpose());
}

TEST(NeuralNetworkTest, fitTest) {
    vector<size_t> size {5,3,6};
    NeuralNetwork<double> nn(size);
    Matrix<double, 5, 1> X;
    X << 5,2,5,2,5;
    Matrix<double, 6, 1> Y;
    Y << 1,1,1,1,1,1;
    nn.loadData(X.transpose(), Y.transpose());

    nn.fit();
}

TEST(NeuralNetworkTest, fitTestMultiLayer) {
    vector<Layer1D<double>> net {Layer1D<double>(5,3,"reLu"), Layer1D<double>(3, 6, "tanH")};
    NeuralNetwork<double> nn(net);
    Matrix<double, 5, 1> X;
    X << 5,2,5,2,5;
    Matrix<double, 6, 1> Y;
    Y << 1,1,1,1,1,1;
    nn.loadData(X.transpose(), Y.transpose());
    nn.fit();
}

TEST(NeuralNetworkTest, csvReaderTest) {
    string mnist_train {"MNIST/mnist_train.csv"};
    vector<vector<double>> train = read_csv(mnist_train);
    ASSERT_EQ(train[1].size(), 785);
    ASSERT_EQ(train.size(), 60000);
}

TEST(NeuralNetworkTest, mnistTest) {
    string train {"MNIST/mnist_train.csv"};
    auto [data, labels]     = prepareData(train);
    for(size_t i = 0; i < data.size(); ++i) for(size_t j = 0; j < 784; ++j) data[i][j] /=255.0;

    vector<size_t> sizes {784,128,10};
    NeuralNetwork<double> nn(sizes);

    nn.loadData(data, labels);
    nn.fit(1);

    string test {"MNIST/mnist_test.csv"};
    auto [data1, labels1] = prepareData(test);
    for(size_t i = 0; i < data1.size(); ++i) for(size_t j = 0; j < 784; ++j) data1[i][j] /=255.0;
    double score = nn.score(data1, labels1);
    cout << score <<endl;
    ASSERT_GE(score, 0.60);
}

TEST(NeuralNetworkTest, mnistTestMultiLayers) {
    string train {"MNIST/mnist_train.csv"};
    auto [data, labels]     = prepareData(train);

    vector<Layer1D<double>> net  {  Layer1D<double>(784,  128, "reLu"),
                                    Layer1D<double>(128,   10, "softmax")};
    NeuralNetwork<double> nn(net);
    for(size_t i = 0; i < data.size(); ++i) for(size_t j = 0; j < 784; ++j) data[i][j] /=255.0;

    nn.loadData(data, labels);
    nn.fit(1);
    string test {"MNIST/mnist_test.csv"};
    auto [data1, labels1] = prepareData(test);
    for(size_t i = 0; i < data1.size(); ++i) for(size_t j = 0; j < 784; ++j) data1[i][j] /=255.0;
    double score = nn.score(data1, labels1);
    cout << "score :" <<score<<endl;
    ASSERT_GE(score, 0.60);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}