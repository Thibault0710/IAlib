#include <gtest/gtest.h>
#include "../utils/vectorOperations.hpp"
#include "../models/KMeans.hpp"
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

TEST(KMeansTest, getClassesTest) {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    KMeans<double> cluster(X, 3);// We instanciate a KMeans class with K = 1
    cluster.fit();

    auto classes = cluster.getClasses();

    ASSERT_EQ(classes(0,0), 0);
    ASSERT_EQ(classes(1,0), 1);
    ASSERT_EQ(classes(2,0), 2);
}

TEST(KMeansTest, getInvertedIndexTest) {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {5.5, 6.5}};
    KMeans<double> cluster(X, 3);// We instanciate a KMeans class with K = 1
    cluster.fit();

    auto invertedIndex = cluster.getIndexInverted();

    ASSERT_EQ(invertedIndex[0][0], 0);
    ASSERT_EQ(invertedIndex[1][0], 1);
    ASSERT_EQ(invertedIndex[2][0], 2);
    ASSERT_EQ(invertedIndex[2][1], 3);
}

TEST(KMeansTest, varianceTest) {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {5.5, 6.5}};
    KMeans<double> cluster(X, 3);// We instanciate a KMeans class with K = 1
    cluster.fit();

    auto variances = cluster.variance();

    ASSERT_NEAR(variances(0,0), 0.0, 0.001);
    ASSERT_NEAR(variances(1,0), 0.0, 0.001);
    ASSERT_NEAR(variances(2,0), 0.125, 0.001);
}

TEST(KMeansTest, inertieTest) {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {5.5, 6.5}};
    KMeans<double> cluster(X, 3);// We instanciate a KMeans class with K = 1
    cluster.fit();

    double inertie = cluster.inertie();

    ASSERT_NEAR(inertie, 0.25, 0.001);
}

TEST(KMeansTest, elbowTest) {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {5.5, 6.5}};
    KMeans<double> cluster(X, 3);// We instanciate a KMeans class with K = 1
    cluster.fit();

    auto elbow = cluster.elbow();
    ASSERT_NEAR(elbow(1), 0.25, 0.001); // for _K = 2
    ASSERT_NEAR(elbow(2), 0.0, 0.001);
}

TEST(KMeansTest, mnistTest) { // Méthode pas concluante
    auto pr = prepareData("MNIST/mnist_train.csv");
    vector<vector<double>> data   = pr.first;
    KMeans<double> clusters(data, 10);
    clusters.fit(125, 100);
    auto classes = clusters.getClasses();
   //for(size_t i = 0; i < 1000; ++i) cout << classes(i,0)<<"   " << distance(pr.second[i].begin(), find_if(pr.second[i].begin(), pr.second[i].end(), [](int x) { return x != 0; })) << endl<<endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}