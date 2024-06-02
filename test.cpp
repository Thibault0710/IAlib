#include "LinearRegression.hpp"
#include "LogisticRegression.hpp"
#include <gtest/gtest.h>
using namespace std;

int main(int argc, char **argv) {
        std::vector<std::vector<double>> X = {{1.0, 2.0}, {2.0, 4.0}, {3.0, 6.0}, {0.0, 6.0}};
        std::vector<double> Y = {0.75, 0.75, 0.75, 0.75};  // Classes : 0 ou 1

        LogisticReg<double> log(X, Y);
        auto ret = log.fitAffineMomentumNesterov();
        cout << ret << endl;
        return 0;
}
