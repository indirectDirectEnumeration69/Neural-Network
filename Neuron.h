#pragma once
#include <vector>
#include <algorithm>
#include <variant>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include <map>
#include <variant>
#include <string>

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
class NeuronBranch {
public:
    std::vector<std::variant<double, std::string>> inputs;
    std::unique_ptr<Neuron> nextNeuron;
    std::vector<std::unique_ptr<NeuronBranch>> branches;

    NeuronBranch(const std::vector<std::variant<double, std::string>>& inputs) {
        this->inputs = inputs;
        nextNeuron = std::make_unique<Neuron>();
        nextNeuron->processInputs(inputs);
    }
    void addBranch(const std::vector<std::variant<double, std::string>>& inputs) {
        branches.emplace_back(std::make_unique<NeuronBranch>(inputs));
    }
};
class NeuronTree {
public:
    class DataRelationClassification {
    public:
        std::map<std::string, int> class_labels;

        void addClassLabel(const std::string& label) {
            class_labels[label] = class_labels.size() + 1;
        }

        int getClassLabel(const std::string& label) {
            return class_labels[label];
        }

        void classifyData(Neuron& neuron, const std::vector<std::variant<double, std::string>>& inputs) {
            neuron.inputs = inputs;
            auto class_label = getClassLabel(std::get<std::string>(inputs.back()));
            neuron.output = static_cast<double>(class_label);
        }
    };
};