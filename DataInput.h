#pragma once
#include <iostream>
#include <vector>
#include <variant>
#include <string>
#include <sstream>

class InputLayer {
public:
    int capacity;
    std::vector<std::variant<double, std::string>> data;
    std::vector<std::variant<double, std::string>> x1, x2, x3;
    std::vector<std::variant<double, std::string>> y1, y2, y3;

    InputLayer() = default;

    explicit InputLayer(int cap) : capacity{ cap } {
        data.reserve(capacity);
    }

    void inputData() {
        std::cout << "[Test] Data Input: " << std::endl;
        while (true) {
            std::string input;
            std::cin >> input;
            std::istringstream iss(input);
            double value;
            if (iss >> value) {
                data.emplace_back(value);
            }
            else {
                data.emplace_back(input);
            }
            if (data.size() >= static_cast<std::size_t>(capacity)) {
                break;
            }
        }
        splitData();
    }

private:
    void splitData() {
        const auto third = data.size() / 3;
        x1.insert(x1.end(), data.begin(), data.begin() + third);
        x2.insert(x2.end(), data.begin() + third, data.begin() + 2 * third);
        x3.insert(x3.end(), data.begin() + 2 * third, data.end());

        const auto half = third / 2;
        y1.insert(y1.end(), x1.begin(), x1.begin() + half);
        y2.insert(y2.end(), x2.begin(), x2.begin() + half);
        y3.insert(y3.end(), x3.begin(), x3.begin() + half);
    }
};