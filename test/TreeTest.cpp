#include "../models/DecisionTree/Tree.hpp"
#include <gtest/gtest.h>

TEST(TreeTest, creationTest) {
    vector<vector<double>> data {{1.0,2.0},{-1.0,2.0},{-5.0,5.0},{2.0,5.0},{-10.0,2.0},{-6.0,5.0},{-9.0, -5.0}, {1.0, 6.0}, {2.0,1.0},{5.0,20.0}, {0.0,1.0}};
    vector<int> labels {1,2,2,1,2,2,2,1, 1,1, 1};
    Tree<double> tr(data, labels);
}

TEST(Tree, mnistTest) {
    vector<vector<double>> train_file = read_csv("MNIST/mnist_train.csv");
    size_t datasize = 1000;
    vector<vector<double>> data(datasize);
    vector<int> labels(datasize);
    
    transform(train_file.begin(), train_file.begin()+datasize,   data.begin(), [](vector<double> vec){ return vector<double>(vec.begin() + 1, vec.end()); });
    transform(train_file.begin(), train_file.begin()+datasize, labels.begin(), [](vector<double> vec){ return (int) vec[0]; });

    Tree<double> tr(data, labels);
    tr.fit();

    // Sur les données de test
    vector<vector<double>> test_file = read_csv("MNIST/mnist_test.csv");
    vector<vector<double>> test_data(1000);
    vector<int> label(1000);
    
    transform(test_file.begin(), test_file.begin()+1000, test_data.begin(), [](vector<double> vec){ return vector<double>(vec.begin() + 1, vec.end()); });
    transform(test_file.begin(), test_file.begin()+1000,     label.begin(), [](vector<double> vec){ return (int) vec[0]; });

    int compt = 0;
    for(size_t i = 0; i < 1000; ++i) if(tr.feedforward(test_data[i]) == label[i]) ++compt;
    ASSERT_GE(compt, 500); // On veut plus de 50% de réussite
}

TEST(TreeTest, irisTest) {
    auto [data, labels_text] = prepareDataIris("Iris/Iris.csv");
    unordered_map<string, int> mp;
    int current = 0;

    for(auto &label : labels_text) if(mp.find(label) == mp.end()) mp[label] = current++;
    vector<int> labels_int(labels_text.size());
    transform(labels_text.begin(), labels_text.end(), labels_int.begin(), [&mp](const string &str){ return mp[str]; });

    Tree<double> tr(data, labels_int);
    tr.fit();

    int compt = 0;
    for(size_t i = 0; i < data.size(); ++i) if(tr.feedforward(data[i]) == labels_int[i]) ++compt;
    ASSERT_GE(compt, 140);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}