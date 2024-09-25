#ifndef GINI_HPP
#define GINI_HPP

#include <bits/stdc++.h>
using namespace std;

double gini(const vector<int> &labels) {
    unordered_map<int, int>mp;
    for(auto &e : labels) ++mp[e];
    return 1.0-(accumulate(mp.begin(), mp.end(), 0.0, [&mp, &labels](double sum, pair<int, int>pr){return sum + pow(((double) pr.second)/labels.size(), 2); }));
}

double gini(const vector<double> &probabilities) {
    return 1.0 - std::accumulate(probabilities.begin(), probabilities.end(), 0.0, [](double sum, double p) { return sum + p * p; });
}

#endif