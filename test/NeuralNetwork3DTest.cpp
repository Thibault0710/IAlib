#include <gtest/gtest.h>
#include "../models/NeuralNetwork/CNN/NeuralNetwork3D.hpp"
#include "../utils/csv.hpp"
#include "../utils/activation.hpp"
using namespace std;

TEST(NeuralNetwork3DTest, constructorTest) {
    vector<triplet<size_t, size_t, size_t>> size {{5,5, 1}, {3,3, 1}, {6,6, 1}};
    NeuralNetwork3D<double> nn(size);
}

TEST(NeuralNetwork3DTest, feedforwardTest) {
    vector<triplet<size_t, size_t, size_t>> size {{1,5, 5}, {3,3, 1}, {1,6, 6}};
    NeuralNetwork3D<double> nn(size);
    Matrix<double, 5, 5> X;
    X << 5, 2, 5, 2, 5,
         5, 5, 0, 2, 6,
         8, 9, 8, 0, 0,
         0, 1, 5, 2, 6,
         7, 4, 6, 5, 4;
    auto Y = nn.feedforward({X});
    ASSERT_EQ(Y[0].rows(), 6);
    ASSERT_EQ(Y[0].cols(), 6);
}

TEST(NeuralNetwork3DTest, errorTest) {
    vector<triplet<size_t, size_t, size_t>> size {{1,5,5}, {1,3,3}, {1, 6,6}};
    NeuralNetwork3D<double> nn(size);
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
    auto error = nn.error({X},{Y});
}

TEST(NeuralNetwork3DTest, errorTest2) {
    vector<triplet<size_t, size_t, size_t>> size {{1,5,5}, {1,3,3}, {1,6,6}};
    NeuralNetwork3D<double> nn(size);
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
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> inputs {{X}, {X}};
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> outputs {{Y}, {Y}};
    auto error = nn.error(inputs, outputs);
}

TEST(NeuralNetwork3DTest, loadDataTest) {
    vector<triplet<size_t, size_t, size_t>> size {{1, 5,5}, {1, 3,3}, {1,6,6}};
    NeuralNetwork3D<double> nn(size);
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
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> inputs {{X}, {X}};
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> outputs {{Y}, {Y}};
    nn.loadData(inputs, outputs);
}

TEST(NeuralNetwork3DTest, fitTest) {
    vector<triplet<size_t, size_t, size_t>> size {{1, 5,5}, {1, 3,3}, {1,6,6}};
    NeuralNetwork3D<double> nn(size);
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
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> inputs {{X}, {X}};
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> outputs {{Y}, {Y}};
    nn.loadData(inputs, outputs);
    nn.fit();
}

TEST(NeuralNetwork3DTest, mnistTest) {
    string train {"MNIST/mnist_train.csv"};
    vector<triplet<size_t, size_t, size_t>> sizes {{1,28, 28},{16,1, 16},{1,10, 1}};
    NeuralNetwork3D<double> nn(sizes);

    auto [datas, lab] = prepareData2D(train, 28, 28, 10, 1);
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> data_3D(datas.size());
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> labs_3D(datas.size());
    for(size_t i = 0; i < datas.size(); ++i) {
        data_3D[i] = {datas[i]};
        labs_3D[i] = {lab[i]};
    }
    datas.clear();
    lab.clear();
    nn.loadData(data_3D, labs_3D);
    nn.fit();
    cout<<nn.feedforward(data_3D[1])[0]<<endl;
    nn.clearData();

    string test {"MNIST/mnist_test.csv"};
    auto [X, Y] = prepareData2D(test, 28, 28, 10, 1);
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> X_3D(X.size());
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> Y_3D(X.size());
    cout<<X.size()<<endl;
    for(size_t i = 0; i < X.size(); ++i) {
        X_3D[i] = {X[i]};
        Y_3D[i] = {Y[i]};
    }
    X.clear();
    Y.clear();
    auto predictions = nn.feedforward(X_3D);
    size_t correct = 0;
    for(size_t i = 0; i < X_3D.size(); ++i) {
        int predicted_label = std::distance(predictions[i][0].data(), std::max_element(predictions[i][0].data(), predictions[i][0].data() + predictions[i][0].size()));
        int true_label = std::distance(Y_3D[i][0].data(), std::max_element(Y_3D[i][0].data(), Y_3D[i][0].data() + Y_3D[i][0].size()));
        if(predicted_label == true_label) {
            correct++;
        }
    }
    cout << correct<<endl;
    double accuracy = static_cast<double>(correct) / predictions.size();
    cout << "Accuracy: " << accuracy << endl;
}
/*
TEST(NeuralNetwork3DTest, mnistTestMultiLayers) {
    string train {"MNIST/mnist_train.csv"};
    auto [data, labels]     = prepareData2D(train, 28, 28, 10, 1);

    vector<Layer2D<double>> net  {  Layer2D<double>({28, 28},  {10, 10}, reLu<double>,    reLuPrime<double>),
                                    Layer2D<double>({10, 10},  {10, 1}, sigmoid<double>, sigmoidPrime<double>)};
    NeuralNetwork2D<double> nn(net);

    nn.loadData(data, labels);
    nn.fit();

    string test {"MNIST/mnist_test.csv"};
    auto pr2 = prepareData2D(test, 28, 28, 10, 1);
    double score = nn.score(pr2.first, pr2.second);
    cout << "score :" <<score<<endl;
}
*/
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}