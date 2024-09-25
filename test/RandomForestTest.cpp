#include "../models/DecisionTree/RandomForest.hpp"
#include <gtest/gtest.h>
#include "../utils/csv.hpp"

TEST(RandomForestTest, creationTest) {
    vector<vector<double>> data {{1.0,2.0},{-1.0,2.0},{-5.0,5.0},{2.0,5.0},{-10.0,2.0},{-6.0,5.0},{-9.0, -5.0}, {1.0, 6.0}, {2.0,1.0},{5.0,20.0}, {0.0,1.0}};
    vector<int> labels {1,2,2,1,2,2,2,1, 1,1, 1};
    RandomForest<double> forest(data, labels, 10);
    forest.fit();
}

TEST(RandomForestTest, predictionTest) {
    vector<vector<double>> data {{1.0,2.0},{-1.0,2.0},{-5.0,5.0},{2.0,5.0},{-10.0,-2.0},{-6.0,-5.0},{9.0, 5.0}, {-1.0, 6.0}, {-2.0,-1.0},{5.0,-20.0}, {0.0,0.0}, {1.0,-2.0}};
    vector<int> labels {1,2,2,1,3,3,1,2,3,4,4,4};
    RandomForest<double> forest(data, labels, 10, 100.0);
    forest.fit();
    size_t compt = 0;
    for(size_t i = 0; i < labels.size(); ++i) if(labels[i]== forest.predict(data[i])) ++compt;
    cout << compt << " / " << labels.size() << endl;
}

TEST(RandomForestTest, mnistTest) {
    vector<vector<double>> train_file = read_csv("MNIST/mnist_train.csv");
    size_t datasize                   = 1000;

    vector<vector<double>> data(datasize);
    vector<int> labels(datasize);
    
    transform(train_file.begin(), train_file.begin()+datasize,   data.begin(), [](vector<double> vec){ return vector<double>(vec.begin() + 1, vec.end()); });
    transform(train_file.begin(), train_file.begin()+datasize, labels.begin(), [](vector<double> vec){ return (int) vec[0]; });

    RandomForest<double> forest(data, labels, 50, 0.3);
    forest.fit();

    // Sur les données de test
    vector<vector<double>> test_file = read_csv("MNIST/mnist_test.csv");
    vector<vector<double>> test_data(1000);
    vector<int> label(1000);
    
    transform(execution::par, test_file.begin(), test_file.begin()+1000, test_data.begin(), [](vector<double> vec){ return vector<double>(vec.begin() + 1, vec.end()); });
    transform(execution::par, test_file.begin(), test_file.begin()+1000,     label.begin(), [](vector<double> vec){ return (int) vec[0]; });

    int compt = 0;
    for(size_t i = 0; i < 1000; ++i) if(forest.predict(test_data[i]) == label[i]) ++compt;
    cout << compt << endl;
    ASSERT_GE(compt, 500); // On veut plus de 50% de réussite
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}