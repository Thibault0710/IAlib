#include "../models/DecisionTree/Node.hpp"
#include <gtest/gtest.h>
#include "../utils/csv.hpp"

TEST(NodeTest, creationTest) {
    Node<double> nd;
    vector<vector<double>> data {{1.0,2.0},{-1.0,2.0},{-5.0,5.0},{2.0,5.0},{-10.0,2.0},{-6.0,5.0},{-9.0, -5.0}, {1.0, 6.0}, {2.0,1.0},{5.0,20.0}, {0.0,1.0}};
    vector<int> labels {1,2,2,1,2,2,2,1, 1,1, 1};
    nd.setThreshold(data, labels);
    ASSERT_EQ(nd.getVariable(), 0);
    ASSERT_NEAR(nd.getThreshold(), 0.0, 0.001);
}

TEST(NodeTest, fitTest) {
    Node<double> nd;
    vector<vector<double>> data {{1.0,2.0},{-1.0,2.0},{-5.0,5.0},{2.0,5.0},{-10.0,-2.0},{-6.0,-5.0},{9.0, 5.0}, {-1.0, 6.0}, {-2.0,-1.0},{5.0,-20.0}, {0.0,0.0}, {1.0,-2.0}};
    vector<int> labels {1,2,2,1,3,3,1,2,3,4,4,4};
    nd.fit(data, labels);
    for(size_t i = 0; i < labels.size(); ++i) ASSERT_EQ(nd.feedforward(data[i]), labels[i]);
}

TEST(NodeTest, mnistTest) {
    vector<vector<double>> train_file = read_csv("MNIST/mnist_train.csv");
    size_t datasize = 1000;
    vector<vector<double>> data(datasize);
    vector<int> labels(datasize);
    
    transform(train_file.begin(), train_file.begin()+datasize,   data.begin(), [](vector<double> vec){ return vector<double>(vec.begin() + 1, vec.end()); });
    transform(train_file.begin(), train_file.begin()+datasize, labels.begin(), [](vector<double> vec){ return (int) vec[0]; });

    Node<double> nd;
    nd.fit(data, labels);

    // Sur les données de test
    vector<vector<double>> test_file = read_csv("MNIST/mnist_test.csv");
    vector<vector<double>> test_data(1000);
    vector<int> label(1000);
    
    transform(test_file.begin(), test_file.begin()+1000, test_data.begin(), [](vector<double> vec){ return vector<double>(vec.begin() + 1, vec.end()); });
    transform(test_file.begin(), test_file.begin()+1000,     label.begin(), [](vector<double> vec){ return (int) vec[0]; });

    int compt = 0;
    for(size_t i = 0; i < 1000; ++i) if(nd.feedforward(test_data[i]) == label[i]) ++compt;
    ASSERT_GE(compt, 500); // On veut plus de 50% de réussite
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}