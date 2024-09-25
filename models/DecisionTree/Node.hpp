#ifndef NODE_HPP
#define NODE_HPP

#include <bits/stdc++.h>
#include <execution>
#include "../../utils/gini.hpp"
#include "../../utils/vectorOperations.hpp"
using namespace std;

template<typename T>
class Node{
private: 
    T _threeshold;
    size_t _variable;
    int _value; // La valeur du noeud dans le cas ou c'est une feuille
    int _maxDepth;
    Node<T> *_left;
    Node<T> *_right;
    static constexpr double _gini_threshold = 0.01;
    static constexpr size_t _gini_min_size  = 2; 

public:
    Node(int maxDepth = 50, T threshold = 0, size_t variable=-1, int value = numeric_limits<int>::max()) : _threeshold(threshold), _variable(variable), _value(value), _maxDepth(maxDepth), _left(nullptr), _right(nullptr) {}

    Node(const Node<T> &nd) : _threeshold(nd._threeshold), _variable(nd._variable), _value(nd._value), _maxDepth(nd._maxDepth), _left(nullptr), _right(nullptr) {
        if(nd._left)  this->_left  = new Node<T>(*nd._left);
        if(nd._right) this->_right = new Node<T>(*nd._right);
    }

    Node<T>& operator=(const Node<T> &nd) {
        if(this != &nd) {
            Node<T> tmp(nd);
            swap(_threeshold, tmp.threeshold);
            swap(_variable, tmp._variable);
            swap(_value, tmp._value);
            swap(_maxDepth, tmp._maxDepth);
            swap(_left, tmp._left);
            swap(_right, tmp._right);
        }
        return *this;
    }

    ~Node() {
        delete _left;
        delete _right;
    }

    T getThreshold() const {
        return this->_threeshold;
    }

    size_t getMaxDepth() const {
        return this->_maxDepth;
    }

    size_t getVariable() const {
        return this->_variable;
    }

    int getValue() const {
        return this->_value;
    }

    pair<vector<size_t>, vector<size_t>> setThreshold(const std::vector<std::vector<T>> &data, const std::vector<int> &labels) {
        // ret.first -> should we continue the tree ?
        // ret.second indices of left nodes
        // ret.third indices of rights nodes
        double initial_gini     = gini(labels);
        double final_gini       = std::numeric_limits<double>::max();
        size_t final_variable   = -1;
        double final_threeshold = -1.0;

        vector<size_t> left_nodes;
        vector<size_t> right_nodes;

        for(size_t variable = 0; variable < data[0].size(); ++variable) {
            for(size_t threeshold_index = 0; threeshold_index < data.size(); ++threeshold_index) {
                vector<int> left;
                vector<int> right;
                vector<size_t> left_idx;
                vector<size_t> right_idx;

                for(size_t i = 0; i < data.size(); ++i) {
                    if (data[i][variable] >= data[threeshold_index][variable]) {
                        right.push_back(labels[i]);
                        right_idx.push_back(i);
                    }
                    else {
                        left.push_back(labels[i]);
                        left_idx.push_back(i);
                    }
                }

                double gini_left  = gini(left);
                double gini_right = gini(right);
//cout << gini_left<<endl;
                double gini_value = gini_left * ((double)left.size())/data.size() + gini_right * ((double)right.size()) / data.size();

                if(gini_value < final_gini) {
                    final_gini       = gini_value;
                    final_variable   = variable;
                    final_threeshold = data[threeshold_index][variable];
                    left_nodes       = left_idx;
                    right_nodes      = right_idx;
                }
            }
        }


        if(initial_gini == final_gini) return {vector<size_t>{}, vector<size_t>{}};
        _threeshold = final_threeshold;
        _variable   = final_variable;
        return {left_nodes, right_nodes};
    }

    void fit(const vector<vector<T>> &X, const vector<int> &Y) {
        auto pr = this->setThreshold(X, Y);
        if(pr.first.size() == 0) return ;
        if(pr.second.size() == 0) return ;

        vector<vector<T>> data_left(pr.first.size());
        vector<vector<T>> data_right(pr.second.size());
        vector<int> label_left(pr.first.size());
        vector<int> label_right(pr.second.size());

        transform(execution::par, pr.first.begin(),  pr.first.end(),  begin(data_left),  [&](size_t index){ return X[index]; });
        transform(execution::par, pr.second.begin(), pr.second.end(), begin(data_right), [&](size_t index){ return X[index]; });

        transform(execution::par, pr.first.begin(),  pr.first.end(),  label_left.begin(),  [&](size_t index){ return Y[index]; });
        transform(execution::par, pr.second.begin(), pr.second.end(), label_right.begin(), [&](size_t index){ return Y[index]; });

        if(this->_maxDepth >= 0 && gini(label_left) > _gini_threshold) {
            _left  = new Node<T>(_maxDepth - 1);
            _left->fit(data_left, label_left);
        } 
        else _left  = new Node<T>( _maxDepth - 1, 0, 0,  mostFrequentValue(label_left));

        if(this->_maxDepth >= 0 && gini(label_right) > _gini_threshold) {
            _right = new Node<T>(_maxDepth - 1);
            _right->fit(data_right, label_right);
        }
        else _right  = new Node<T>(_maxDepth - 1, 0, 0,  mostFrequentValue(label_right));
    }

    void print(int depth = 0) const {
        for(int i = 0; i < depth; ++i) cout << "  ";
        cout << "Node(threshold: " << _threeshold << ", variable: " << _variable << ", value: " <<_value << ")\n";
        if(_left)  _left->print(depth + 1);
        if(_right) _right->print(depth + 1);
    }

    int feedforward(const vector<double> &data) {
        if(_value != numeric_limits<int>::max()) return _value;
        if(_right && data[_variable] >= _threeshold) return _right->feedforward(data);
        if(_left  && data[_variable] <  _threeshold) return  _left->feedforward(data);
        return numeric_limits<int>::max();
    }
};

#endif