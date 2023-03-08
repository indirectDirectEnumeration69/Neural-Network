#pragma once
#include <vector>
#include <algorithm>
#include <variant>
#include <string>
#include <sstream>

class Neuron {
public:
    std::vector<std::variant<double, std::string>> inputs;
    std::vector<double> weights;
    std::vector<double> biases;
    double output;

    Neuron() = default;

    void processInputs(const std::vector<std::variant<double, std::string>>& inputs) {
        this->inputs = inputs;
        const auto weightedSum = calculateWeightedSum();
        this->output = activationFunction(weightedSum);
    }

private:
    double calculateWeightedSum() const {
        double result = 0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            auto input = std::get_if<double>(&inputs[i]);
            if (!input) {
                continue;
            }
            result += *input * weights[i];
        }
        result += biases[0];
        return result;
    }

    double activationFunction(double weightedSum) const {
        return std::max(0.0, weightedSum);
    }
};