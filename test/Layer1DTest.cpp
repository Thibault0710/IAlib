#include <gtest/gtest.h>
#include "../models/NeuralNetwork/Layer1D.hpp"
#include "../utils/activation.hpp"
using namespace std;

TEST(Layer1DTest, initTest) {
    Layer1D<double> layer(5, 7);
    auto w    = layer.getWeights();
    auto b    = layer.getBias();
    auto mean = w.sum() / 35;
    ASSERT_GE(mean, -0.329610159);
    ASSERT_LE(mean, 0.329610159);//interval de confiance à 95% pour la moyenne
}

TEST(Layer1DTest, feedforwardTestsigmoid) {
    Layer1D<double> layer(5, 7);
    auto w    = layer.getWeights();
    auto b    = layer.getBias();
    Matrix<double, 5, 1> input;
    input << 1,1,1,1,1;
    auto f    = layer.feedforward(input);
}

TEST(Layer1DTest, feedforwardTestreLu) {
    Layer1D<double> layer(5, 7, reLu<double>, reLuPrime<double>);
    auto w    = layer.getWeights();
    auto b    = layer.getBias();
    Matrix<double, 5, 1> input;
    input << 1,1,1,1,1;
    auto f    = layer.feedforward(input);
}

TEST(Layer1DTest, feedforwardTesttanH) {
    Layer1D<double> layer(5, 7, tanH<double>, tanHPrime<double>);
    auto w    = layer.getWeights();
    auto b    = layer.getBias();
    Matrix<double, 5, 1> input;
    input << 1,1,1,1,1;
    auto f    = layer.feedforward(input);
}

TEST(Layer1DTest, feedforwardTestSoftplus) {
    Layer1D<double> layer(5, 7, softplus<double>, softplusPrime<double>);
    auto w    = layer.getWeights();
    auto b    = layer.getBias();
    Matrix<double, 5, 1> input;
    input << 1,1,1,1,1;
    auto f    = layer.feedforward(input);
}

TEST(Layer1DTest, errorTest) {
    Layer1D<double> layer(5, 7);
    auto w    = layer.getWeights();
    auto b    = layer.getBias();
    Matrix<double, 5, 1> input;
    input << 1,1,1,1,1;
    Matrix<double, 7, 1> output;
    output << 1,1,1,1,1,1,1;
    auto f    = layer.error(input, output);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}