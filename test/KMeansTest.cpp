#include <gtest/gtest.h>
#include "../utils/vectorOperations.hpp"
#include "../KMeans.hpp"
using namespace std;

TEST(KMeansTest, constructorTest) {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    KMeans<double> cluster(X, 1);// We instanciate a KMeans class with K = 1
    cluster.fit();

    ASSERT_NEAR(cluster.getData()(0,0), 1.0, 0.001);
    ASSERT_NEAR(cluster.getData()(0,1), 2.0, 0.001);
}

TEST(KMeansTest, fitMethodTest) {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    KMeans<double> cluster(X, 1);// We instanciate a KMeans class with K = 1
    cluster.fit();

    auto centroids = cluster.getCentroids();

    ASSERT_NEAR(centroids(0,0), 3.0, 0.001);
    ASSERT_NEAR(centroids(0,1), 4.0, 0.001);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}