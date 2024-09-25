#include <gtest/gtest.h>
#include "../models/NeuralNetwork/RNN/LayerRNN.hpp"

TEST(LayerRNNTest, constructor) {
    LayerRNN<double>rnn;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}