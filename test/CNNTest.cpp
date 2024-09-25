#include <gtest/gtest.h>
#include "../models/NeuralNetwork/CNN/CNN.hpp"

TEST(CNNTest, creation) {
    CNN<double>cnn;
}

TEST(CNNTest, feedforward) {
    vector<variant<LayerConv<double>, Layer3D<double>>> net;
    net.push_back(LayerConv<double>({3,3}, {1,5,5}));
    net.push_back(Layer3D<double>({64,3,3}, {1,6,6}));
    CNN<double> nn(net);
    Matrix<double, 5, 5> X;
    X << 5, 2, 5, 2, 5,
         5, 5, 0, 2, 6,
         8, 9, 8, 0, 0,
         0, 1, 5, 2, 6,
         7, 4, 6, 5, 4;
    auto Y = nn.feedforward({X});
    ASSERT_EQ(Y[0].rows(), 6);
    ASSERT_EQ(Y[0].cols(), 6);
    ASSERT_EQ(Y.size(), 1);
}

TEST(CNNTest, error) {
    vector<variant<LayerConv<double>, Layer3D<double>>> net;
    net.push_back(LayerConv<double>({3,3}, {1,5,5}));
    net.push_back(Layer3D<double>({64,3,3}, {1,6,6}));
    CNN<double> nn(net);
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

TEST(CNNTest, error2) {
    vector<variant<LayerConv<double>, Layer3D<double>>> net;
    net.push_back(LayerConv<double>({3,3}, {1,5,5}, 1));
    net.push_back(Layer3D<double>({1,3,3}, {1,6,6}));
    CNN<double> nn(net);
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

TEST(CNNTest, loadData) {
    vector<variant<LayerConv<double>, Layer3D<double>>> net;
    net.push_back(LayerConv<double>({3,3}, {1,5,5}, 1));
    net.push_back(Layer3D<double>({1,3,3}, {1,6,6}));
    CNN<double> nn(net);    
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

TEST(CNNTest, fitLayer3D) {
    vector<variant<LayerConv<double>, Layer3D<double>>> net;
    net.emplace_back(Layer3D<double>({1,5,5}, {64,3,3}));
    net.emplace_back(Layer3D<double>({64,3,3}, {1,6,6}));
    CNN<double> nn(net); 
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

TEST(CNNTest, fitBoth) {
    vector<variant<LayerConv<double>, Layer3D<double>>> net;
    net.emplace_back(LayerConv<double>({3,3}, {1,5,5}));
    net.emplace_back(Layer3D<double>({64,3,3}, {1,6,6}));
    CNN<double> nn(net); 
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
    vector<variant<LayerConv<double>, Layer3D<double>>> net;
    net.emplace_back(Layer3D<double>({1,28,28}, {3,10,10}));
    net.emplace_back(LayerConv<double>({3, 3}, {3,10,10}));
    net.emplace_back(Layer3D<double>({3*64,8,8}, {1,10,1}));
    CNN<double> nn(net); 

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
    nn.fit(1);
    nn.clearData();

    string test {"MNIST/mnist_test.csv"};
    auto [X, Y] = prepareData2D(test, 28, 28, 10, 1);
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> X_3D(X.size());
    vector<vector<Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>> Y_3D(X.size());
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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}