#include <gtest/gtest.h>
#include "../KNN.hpp"
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
    auto res = knn.fit(X_test);
    cout << res <<endl;

    ASSERT_NEAR(knn.getData()(0, 0), 1.0, 0.001);
    ASSERT_NEAR(knn.getData()(0, 1), 2.0, 0.001);
}

int main(int argc, char**argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}