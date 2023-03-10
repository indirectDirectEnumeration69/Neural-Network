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
    NeuronTree tree;
    NeuronTree::DataRelationClassification classification;  
    classification.addClassLabel("ClassRelationOne");
    classification.addClassLabel("ClassRelationTwo");
    classification.addClassLabel("ClassRelationThree");
    for (auto& neuron : hidden_layer->neurons) {
        classification.classifyData(neuron, input_layer->data);
    }
    for (auto& neuron : hidden_layer->neurons) {
        neuron.processInputs(input_layer->data);
    }
    std::vector<double> outputs_from_hidden_layer;
    for (const auto& neuron : hidden_layer->neurons) {
        outputs_from_hidden_layer.push_back(neuron.output);
    }
    auto rootBranch = std::make_unique<NeuronBranch>(outputs_from_hidden_layer);

    rootBranch->addBranch({ outputs_from_hidden_layer[0] * 2, outputs_from_hidden_layer[1] * 4, outputs_from_hidden_layer[2] * 6 });
    rootBranch->addBranch({ outputs_from_hidden_layer[0] * 3, outputs_from_hidden_layer[1] * 5, outputs_from_hidden_layer[2] * 7 });
    
    switch (rootBranch->branches.size()) {
    case 2:
        if (hidden_layer->neurons.size() <= 3) {
            int current_size = hidden_layer->neurons.size()+1;
            int target_size = current_size*3;
            hidden_layer->neurons.reserve(target_size);
            for (int i = current_size; i < target_size; ++i) {
                hidden_layer->neurons.emplace_back(Neuron{});
                std::vector<double> concat_weights;
                for (int j = 0; j <= 3; ++j) {
                    concat_weights.insert(concat_weights.end(), hidden_layer->neurons[j].weights.begin(), hidden_layer->neurons[j].weights.end());
                }
                hidden_layer->neurons[i].weights.assign(concat_weights.begin(), concat_weights.end()); 
            }
            auto branch = std::make_unique<NeuronBranch>();
            int i = 3;
            rootBranch->addBranch({ outputs_from_hidden_layer[i] * 4 });
        }
       else if(hidden_layer->neurons.size() > 3) {
            do {
                hidden_layer->neurons.emplace_back(Neuron{});
            } while (hidden_layer->neurons.size() <= 10);
        }
        
        
        std::cout << "Branches added successfully" << std::endl;
        break;



    default: 
            std::cout << "Branches not added successfully" << std::endl;
            break;
    }
    auto output_layer = std::make_unique<OutputLayer>();
    output_layer->weights = { 0.2, 0.3, 0.5 };
    output_layer->biases = { 0.2 };

    std::vector<std::variant<double, std::string>> outputs_from_hidden_layer_var;
    for (const auto& output : outputs_from_hidden_layer) {
        outputs_from_hidden_layer_var.push_back(output);
    }
    output_layer->processInputs(outputs_from_hidden_layer_var);

    std::cout << "[Test] output: " << output_layer->output << std::endl;

    return 0;
}