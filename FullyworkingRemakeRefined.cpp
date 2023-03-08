#include <iostream>
#include "DataInput.h"
#include "Neuron.h"
#include "HiddenLayer.h"
#include "OutputLayer.h"

int main() {
	auto input_layer = std::make_unique<InputLayer>();
	input_layer->inputData();
	auto hidden_layer = std::make_unique<HiddenLayer>();
	hidden_layer->neurons.emplace_back(Neuron{});
	hidden_layer->neurons.emplace_back(Neuron{});
	hidden_layer->neurons.emplace_back(Neuron{});

	hidden_layer->neurons[0].weights = { 0.5, 0.1, 0.2 };
	hidden_layer->neurons[0].biases = { 0.3 };
	hidden_layer->neurons[1].weights = { 0.3, 0.2, 0.4 };
	hidden_layer->neurons[1].biases = { 0.1 };
	hidden_layer->neurons[2].weights = { 0.2, 0.3, 0.5 };
	hidden_layer->neurons[2].biases = { 0.2 };

	for (auto& neuron : hidden_layer->neurons) {
		neuron.processInputs(input_layer->data);
	}
	auto output_layer = std::make_unique<OutputLayer>();
	output_layer->weights = { 0.2, 0.3, 0.5 };
	output_layer->biases = { 0.2 };

	std::vector<std::variant<double, std::string>> outputs_from_hidden_layer;
	for (const auto& neuron : hidden_layer->neurons) {
		outputs_from_hidden_layer.push_back(neuron.output);
	}

	output_layer->processInputs(outputs_from_hidden_layer);

	std::cout << "[Test] output: " << output_layer->output << std::endl;

	return 0;
}