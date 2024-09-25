#include <gtest/gtest.h>
#include "../utils/gini.hpp"
using namespace std;

TEST(GiniTest, gini) {
    vector<int>labels {1,2,2,2,2,2,10,10,10,10};
    ASSERT_NEAR(gini(labels), 0.58, 0.001);
}

int main(int argc, char**argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}