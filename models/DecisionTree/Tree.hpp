#ifndef TREE_HPP
#define TREE_HPP

#include "Node.hpp"

template<typename T>
class Tree {
private:
    Node<T> *root;
    vector<vector<T>> _X; // les données
    vector<int> _Y; //les labels
    int _maxDepth;

public:
    Tree(const vector<vector<T>> &X, const vector<int> &Y, int maxDepth=50) : root(nullptr), _X(X), _Y(Y), _maxDepth(maxDepth) {}

    ~Tree() {
        delete root;
    }

    void fit() {
        root = new Node<T>(_maxDepth);
        root->fit(_X, _Y);
    }

    int feedforward(const vector<double> &data) {
        return root->feedforward(data);
    }

    vector<int> feedforward(const vector<vector<double>> &data) {
        vector<int> ret(data.size());
        transform(data.begin(), data.end(), ret.begin(), [this](vector<double> vec){return this->feedforward(vec);});
        return ret;
    }

    void print() {
        root->print();
    }
};

#endif