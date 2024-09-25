#include <gtest/gtest.h>
#include "../models/NeuralNetwork/Layer2D.hpp"

TEST(Layer2DTest, creationTest) {
    Layer2D<double> lay({28,28},{16,16});
    Matrix<double, 28,28> input;
    input.setZero();
    lay.feedforward(input);
    lay.feedforwardNoApply(input);
}

TEST(Layer2DTest, errorTest) {
    Layer2D<double> lay({28,28},{16,16});
    
    Matrix<double, 28,28> input;
    Matrix<double, 16, 16> output;
    input.setZero();
    output.setZero();

    lay.error(input, output);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}