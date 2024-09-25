#include <gtest/gtest.h>
#include "../models/AutoEncoder.hpp"

TEST(AutoEncoderTest, constructor) {
    AutoEncoder<double> encode(5, 2);
}

TEST(AutoEncoderTest, loadData) {
    AutoEncoder<double> encode(5, 2);
    Matrix<double, Dynamic, Dynamic> X(5, 5);
    X << 1,1,1,1,1,
         2,2,2,2,2,
         1,10,2,3,6,
         3,1,4,5,4,
         2,2,0,2,1;
    encode.loadData(X);
}

TEST(AutoEncoderTest, fit) {
    AutoEncoder<double> encode(5, 2);
    Matrix<double, Dynamic, Dynamic> X(5, 5);
    X << 1,1,1,1,1,
         2,2,2,2,2,
         1,10,2,3,6,
         3,1,4,5,4,
         2,2,0,2,1;
    encode.loadData(X);
    encode.fit();
}

TEST(AutoEncoderTest, feedforward) {
    AutoEncoder<double> encode(5, 2);
    Matrix<double, Dynamic, Dynamic> X(6, 5);
    X << 1,18787,1878,1889,14411,
         2,2000,102,2000,2000,
         1,1000,200,300,600,
         1,401,404,5022,404,
         2,85542,408787,2.022,1.25,
         1,56,247,889,2114;
    encode.loadData(X);
   // encode.fit();
 //   encode.fit();
    for(size_t i = 0; i < 6; ++i) cout << encode.feedforward(X.row(3).transpose()) << endl<<endl;
}

TEST(NeuralNetworkTest, mnist) {
    string train {"MNIST/mnist_train.csv"};
    auto [data, labels] = prepareData(train);
    for(size_t i = 0; i < data.size(); ++i) for(size_t j = 0; j < 256; ++j) data[i][j] /=255.0;

    AutoEncoder<double> encode(784, 2);
    //for(auto &e : data) {
   //     for(auto &ee : e) cout << ee <<" "; 
     //   cout << endl;}
    encode.loadData(data);
    encode.fit();

    for(size_t i = 0; i < data.size(); ++i) cout << encode.feedforward(data[i]) << endl<<endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}