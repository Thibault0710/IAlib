#include "../models/LinearRegression.hpp"
#include "../utils/vectorOperations.hpp"
#include <gtest/gtest.h>
using namespace std;

TEST(RegLinTest, LinearRegressionCalculation1) {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    vector<double> Y         = {3.0, 7.0, 11.0};

    RegLin<double> reg(X, Y);
    auto coefficients = reg.fit();

    ASSERT_NEAR(coefficients(0), 1.0, 0.001);
    ASSERT_NEAR(coefficients(1), 1.0, 0.001);
}

TEST(RegLinTest, LinearRegressionCalculation2) {
    vector<vector<double>> X = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {0.0, 6.0}};
    vector<double> Y         = {3.0, 6.0, 9.0, 6.0};

    RegLin<double> reg(X, Y);
    auto coefficients = reg.fit();

    ASSERT_NEAR(coefficients(0), 1.0, 0.001);
    ASSERT_NEAR(coefficients(1), 1.0, 0.001);

    auto coefs = reg.getCoefs();
    ASSERT_NEAR(coefs(0), coefficients(0), 0.001);
    ASSERT_NEAR(coefs(1), coefficients(1), 0.001);

    auto feed = reg.feedforward(X);
    ASSERT_NEAR(feed(0), Y[0], 0.001);
    ASSERT_NEAR(feed(1), Y[1], 0.001);
    ASSERT_NEAR(feed(2), Y[2], 0.001);
    ASSERT_NEAR(feed(3), Y[3], 0.001);
}

TEST(RegLinTest, LinearRegressionCalculation3) {
        vector<vector<double>> X = {{1.0}, {2.0}, {6.0}, {0.0}, {10000}};
        vector<double> Y         = {5.0, 5.0, 5.0, 5.0, 5.0};

        RegLin<double> reg(X, Y);
        auto coefficients = reg.fit();

        ASSERT_NEAR(coefficients(0), 0, 0.001);
}

TEST(RegLinTest, AffineRegressionCalculation1) {
        vector<vector<double>> X = {{1.0}, {2.0}, {6.0}, {0.0}, {10000}};
        vector<double> Y         = {5.0, 5.0, 5.0, 5.0, 5.0};

        RegLin<double> reg(X, Y);
        auto coefficients = reg.fitAffine();

        ASSERT_NEAR(coefficients(0), 0, 0.001);
        ASSERT_NEAR(coefficients(1), 5, 0.001);
}

TEST(RegLinTest, AffineRegressionCalculation2) {
        vector<vector<double>> X = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {0.0, 6.0}};
        vector<double> Y         = {3.0, 6.0, 9.0, 6.0};

        RegLin<double> reg(X, Y);
        auto coefficients = reg.fitAffine();

        ASSERT_NEAR(coefficients(0), 1.0, 0.001);
        ASSERT_NEAR(coefficients(1), 1.0, 0.001);
        ASSERT_NEAR(coefficients(2), 0.0, 0.001);
}

TEST(RegLinTest, EqualFeaturesNumberNonConst) {
        vector<vector<double>> X = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {0.0, 6.0}};
        vector<double> Y         = {3.0, 6.0, 9.0, 6.0};
        RegLin<double> reg(X, Y);

        ASSERT_EQ(reg.getFeaturesNumber(), 2);
}

TEST(RegLinTest, EqualFeaturesNumberConst) {
        vector<vector<double>> X = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {0.0, 6.0}};
        vector<double> Y         = {3.0, 6.0, 9.0, 6.0};
        RegLin<double> const reg(X, Y);

        ASSERT_EQ(reg.getFeaturesNumber(), 2);
}

TEST(RegLinTest, SameDataNonConst) {
        vector<vector<double>> X = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {0.0, 6.0}};
        vector<double> Y         = {3.0, 6.0, 9.0, 6.0};
        RegLin<double> reg(X, Y);

        ASSERT_TRUE(matricesAreEqual(reg.getData(), X));
}

TEST(RegLinTest, SameDataConst) {
        vector<vector<double>> X = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {0.0, 6.0}};
        vector<double> Y         = {3.0, 6.0, 9.0, 6.0};
        RegLin<double> const reg(X, Y);

        ASSERT_TRUE(matricesAreEqual(reg.getData(), X));
}

TEST(RegLinTest, SameValuesNonConst) {
        vector<vector<double>> X = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {0.0, 6.0}};
        vector<double> Y         = {3.0, 6.0, 9.0, 6.0};
        RegLin<double> reg(X, Y);

        ASSERT_TRUE(matricesAreEqual(reg.getValues(), toOneVector<double>(Y)));
}

TEST(RegLinTest, SameValuesConst) {
        vector<vector<double>> X = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {0.0, 6.0}};
        vector<double> Y         = {3.0, 6.0, 9.0, 6.0};
        RegLin<double> const reg(X, Y);

        ASSERT_TRUE(matricesAreEqual(reg.getValues(), toOneVector<double>(Y)));
}

TEST(RegLinTest, UnequalNumberOfRows) {
    vector<vector<double>> X = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    vector<double> Y = {3.0, 7.0};

    ASSERT_THROW(RegLin<double> reg(X, Y), std::invalid_argument);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
