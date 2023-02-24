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
class InputHandle : public NeuralNetwork<T> {
public:
    class RawInput {
        std::unique_ptr<std::vector<T>> x1;
        std::unique_ptr<std::vector<T>> x2;
        std::unique_ptr<std::vector<T>> x3;
        std::unique_ptr<std::vector<T>> rawData;
    public:
        RawInput() {
            rawData = std::make_unique<std::vector<T>>();
            T dataRaw{};
            while (std::cin >> dataRaw) {
                rawData->push_back(dataRaw);
            }
        }
        void Input(std::vector<T>& out_x1, std::vector<T>& out_x2, std::vector<T>& out_x3) {
            out_x1.clear();
            out_x2.clear();
            out_x3.clear();

            for (int i = 0; i < rawData->size(); i++) {
                if (i % 3 == 0) {
                    out_x1.push_back(rawData->at(i));
                }
                else if (i % 3 == 1) {
                    out_x2.push_back(rawData->at(i));
                }
                else if (i % 3 == 2) {
                    out_x3.push_back(rawData->at(i));
                }
            }
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
