#include <gtest/gtest.h>
#include "../models/KNN.hpp"
using namespace std;

TEST(KNNTest, constructorTest) {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    vector<int> Y         = {3, 7, 11};
    KNN<double> knn(X, Y, 1);

    ASSERT_NEAR(knn.getData()(0, 0), 1.0, 0.001);
    ASSERT_NEAR(knn.getData()(0, 1), 2.0, 0.001);
}

TEST(KNNTest, fitTest) {
    vector<vector<double>> X_train = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    vector<int> Y_train            = {2, 1, 2};

    vector<vector<double>> X_test = {{1.2,2.2}};

    KNN<double> knn(X_train, Y_train, 1);
    auto res = knn.feedforward(X_test);

    ASSERT_NEAR(knn.getData()(0, 0), 1.0, 0.001);
    ASSERT_NEAR(knn.getData()(0, 1), 2.0, 0.001);
}

TEST(KNNTest, mnistTest) {
    auto pr = prepareData("MNIST/mnist_train.csv");
    vector<vector<double>> data   = pr.first;
    vector<vector<double>> labels = pr.second;
    vector<int> label(labels.size());
    transform(labels.begin(), labels.end(), label.begin(), [](vector<double> vec) {return distance(vec.begin(),  std::find_if(vec.begin(), vec.end(), [](int x) { return x != 0; })); } );
    KNN<double> knn(data, label, 20);


    string test {"MNIST/mnist_test.csv"};
    auto pr2 = prepareData(test);
    auto res = knn.feedforward(pr2.first);
}

int main(int argc, char**argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}