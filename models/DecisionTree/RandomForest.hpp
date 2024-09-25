#ifndef RANDOM_FOREST_HPP
#define RANDOM_FOREST_HPP

#include "Tree.hpp"
#include "../../utils/vectorOperations.hpp"

template<typename T>
class RandomForest {
private: 
    vector<Tree<T>> _trees;
public:
    RandomForest(const vector<vector<T>> &data, const vector<int> &labels, size_t number_of_trees = 50, double percent_of_samples_per_tree = 0.75) {
        pair<vector<vector<vector<T>>>, vector<vector<int>>> bootstrap = bootstrapSamples<T>(data, labels, number_of_trees, (int) data.size()*percent_of_samples_per_tree);
        for(size_t i = 0; i < number_of_trees; ++i) _trees.emplace_back(bootstrap.first[i], bootstrap.second[i]);
    }

    void fit() {
        for(size_t i = 0; i < _trees.size(); ++i) _trees[i].fit();
    }

    int predict(const vector<T> &data) {
        unordered_map<int, int> mp;
        for(size_t i = 0; i < _trees.size(); ++i) ++mp[_trees[i].feedforward(data)];
        return max_element(mp.begin(), mp.end(), [](pair<int, int> mp_a, pair<int, int> mp_b){ return mp_a.second < mp_b.second; })->first;
    }
};

#endif
