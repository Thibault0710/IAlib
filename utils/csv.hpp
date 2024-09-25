#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

std::vector<std::vector<double>> read_csv(const std::string& filename) {
    std::vector<std::vector<double>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Erreur lors de l'ouverture du fichier " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);

        std::string cell;
        while (std::getline(ss, cell, ','))row.push_back(std::stod(cell));
        data.push_back(row);
    }

    file.close();
    return data;
}

std::vector<std::vector<std::string>> read_csv_string(const std::string &filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    
    std::string line;
    while(std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream ss(line);

        std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(cell);
        data.push_back(row);
    }

    file.close();
    return data;
}

