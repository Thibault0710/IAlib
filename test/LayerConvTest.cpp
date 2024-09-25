#include "../models/NeuralNetwork/CNN/LayerConv.hpp"
#include <gtest/gtest.h>

TEST(LayerConvTest, modify_paddingTest) {
    Matrix<double, Eigen::Dynamic, -1> mat(5,5);
    mat.setOnes();
    modify_padding(mat, 2);
}

TEST(LayerConvTest, defaultConstructorTest) {
    LayerConv<double> ly;
}

TEST(LayerConvTest, constructorTest) {
    LayerConv<double> ly({3, 3}, {1,28, 28});
    Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X(28,28);
    X.setConstant(1.0);
    ly.getFilter(63);
    ly.getFilters();
    ly.setFilterConstant(0.0,1);
    auto Y = ly.applyFilter(X, 1);
    cout << Y<<endl;
    ASSERT_EQ(Y.cols(), 26);
    ASSERT_EQ(Y.rows(), 26);
}

TEST(LayerConvTest, feedforwardTest) {
    LayerConv<double> ly({3, 3}, {1,28, 28});
    Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X(28, 28);
    X.setConstant(1.0);
    auto Y = ly.feedforward(X);
    ASSERT_EQ(Y.size(), 64);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}