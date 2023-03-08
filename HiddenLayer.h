#pragma once
#include <vector>
#include "Neuron.h"
#include <variant>

class HiddenLayer {
public:
    std::vector<Neuron> neurons;

    HiddenLayer() = default;

    explicit HiddenLayer(int numNeurons, int numInputsPerNeuron) {
        neurons.resize(numNeurons);
        for (auto& neuron : neurons) {
            neuron.weights.resize(numInputsPerNeuron);
            neuron.biases.resize(1);
        }
    }

    void processInputs(const std::vector<std::variant<double, std::string>>& inputs) {
        for (auto& neuron : neurons) {
            neuron.processInputs(inputs);
        }
    }

    std::vector<double> getOutputs() const {
        std::vector<double> outputs;
        outputs.reserve(neurons.size());
        for (const auto& neuron : neurons) {
            outputs.push_back(neuron.output);
        }
        return outputs;
    }
};