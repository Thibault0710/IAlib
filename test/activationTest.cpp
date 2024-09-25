#include <gtest/gtest.h>
#include "../utils/activation.hpp"
using namespace std;

TEST(activationTest, get2DFunctionTest) {
    get2DFunction<double>("sigmoid");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}