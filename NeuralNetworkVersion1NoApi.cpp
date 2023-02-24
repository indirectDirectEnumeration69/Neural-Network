// NeuralNetworkVersion1NoApi.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <vector>
#include <memory>
#include <concurrent_unordered_set.h>
#include <concurrent_vector.h>
#include <concurrent_queue.h>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <thread>
#include <future>
#include <chrono>
#include <mutex>

template <typename T>
class NeuralNetwork {
public:
    virtual T Input() = 0;
    virtual T InputHandler() = 0;
    virtual T LayerGeneration() = 0;
    virtual T LayerOrganiser() = 0;
    virtual T CheckOrganisation() = 0;
    virtual T NeuronGeneration() = 0;
};

template <typename T>
class Layer : public NeuralNetwork<T> {
public:
    void LayerGeneration() override {
    }

    void LayerOrganiser() override {
    }

    void CheckOrganisation() override {
    }
};

template <typename T>
class Neuron : public NeuralNetwork<T> {
public:
    void CheckOrganisation() override {
    }

    void NeuronGeneration() override {
    }
};
template <typename T>
class NodeMain {
public:
    struct NodeSystem {
  

        struct NodeTypes {
            struct LayerNode {
                int layerId;
                std::vector<int> neuronIds;
            };

            struct NeuronNode {
                int neuronId;
                std::vector<int> inputIds;
                double weight;
                double bias;
            };
            std::vector<NodeTypes::LayerNode> LayersNode;
            std::vector<NodeTypes::NeuronNode> NeuronsNode;
        };
    };
};

template <typename T>
class InputHandle : public NeuralNetwork<T> {
public:
    class RawInput {
        std::vector<T>* x1;
        std::vector<T>* x2;
        std::vector<T>* x3;
        std::vector<T>* rawData;
    public:
        RawInput():
         x1(new std::vector<T>()),
         x2(new std::vector<T>()), 
         x3(new std::vector<T>()),rawData(new std::vector<T>()) {
            T dataRaw{};
            while (std::cin >> dataRaw) {
                rawData->push_back(dataRaw);
            }
            x1->reserve(rawData->size() / 3 + 1);
            x2->reserve(rawData->size() / 3 + 1);
            x3->reserve(rawData->size() / 3 + 1);
            for (int i = 0; i < rawData->size(); i += 3) {
                x1->push_back((*rawData)[i]);
                if (i + 1 < rawData->size()) {
                    x2->push_back((*rawData)[i + 1]);
                }
                if (i + 2 < rawData->size()) {
                    x3->push_back((*rawData)[i + 2]);
                }
            }
        }

        T Input(std::vector<T>& x1_out, std::vector<T>& x2_out, std::vector<T>& x3_out) {
            x1_out.reserve(x1->size());
            x2_out.reserve(x2->size());
            x3_out.reserve(x3->size());
            x1_out = *x1;
            x2_out = *x2;
            x3_out = *x3;

            return T();
        }

        ~RawInput() {
            delete x1;
            delete x2;
            delete x3;
            delete rawData;
        }
    };
    T Input() override {
        return T();
    }
    T InputHandler() override {
        RawInput rawInput{};
        std::vector<T> x1, x2, x3;
        rawInput.Input(x1, x2, x3);
        return T();
    }
    enum class ChangeHandle {
        Passive,
        Active,
        Agressive
    };
    enum class Status {
        Ready,
        NotReady,
        Error,
        Unknown
    };
    InputHandle(Status change, ChangeHandle handleChanged) {
        switch (change) {
        case Status::Ready: {
            RawInput rawInput{};
            std::vector<T> x1, x2, x3;
            rawInput.Input(x1, x2, x3);
            break;
        }
        case Status::NotReady:
            break;
        default:
            break;
        }
        switch (handleChanged) {
        case ChangeHandle::Passive:
            break;
        case ChangeHandle::Active:
            break;
        case ChangeHandle::Agressive:
            break;
        default:
            handleChanged = ChangeHandle::Active;
            break;
        }
    }
};
template <typename T>
class IntInputHandle : public InputHandle<T> {
public:
    IntInputHandle(typename InputHandle<T>::Status change, typename InputHandle<T>::ChangeHandle handleChanged):InputHandle<T>(change, handleChanged) {}
    T LayerGeneration() override {
        return T();
    }

    T LayerOrganiser() override {
        return T();
    }

    T CheckOrganisation() override {
        return T();
    }

    T NeuronGeneration() override {
        return T();
    }

    T Input() override {
        

        return T();
    }

    T InputHandler() override {
        return T();
    }
};
int main()
{
    IntInputHandle<int>* inputHandle = new IntInputHandle<int>(InputHandle<int>::Status::Ready, InputHandle<int>::ChangeHandle::Active);

    return 0;
}
