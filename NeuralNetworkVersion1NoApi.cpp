// NeuralNetworkVersion1NoApi.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <iostream>
#include <vector>
#include <math.h>
#include <thread>
#include <mutex>
#include <future>
#include <concurrent_unordered_set.h>
#include <sstream>
#include <concurrent_vector.h>
#include <concurrent_queue.h>
template <typename T>
class NeuralNetwork {
public:
    virtual void Input() = 0;
    virtual void InputHandler() = 0;
    virtual void LayerGeneration() = 0;
    virtual void LayerOrganiser() = 0;
    virtual void CheckOrganisation() = 0;
    virtual void NeuronGeneration() = 0;
};

template <typename T>
class InputHandle : public NeuralNetwork<T> {
    class RawInput : public NeuralNetwork<T> {
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
        void Input() override {
            x1 = std::make_unique<std::vector<T>>();
            x2 = std::make_unique<std::vector<T>>();
            x3 = std::make_unique<std::vector<T>>();
            for (int i = 0; i < rawData->size(); i++) {
                if (i % 3 == 0) {
                    x1->push_back(rawData->at(i));
                }
                else if (i % 3 == 1) {
                    x2->push_back(rawData->at(i));
                }
                else if (i % 3 == 2) {
                    x3->push_back(rawData->at(i));
                }
            }
        }
    };
    void InputHandler() override {
     
    }
    enum changeHandle {
        Passive,
        Active,
        Agressive
    };
    enum Status {
        Ready,
        NotReady,
        Error,
        Unknown
    };

    InputHandle(Status change, changeHandle HandleChanged) {
        switch (change) {
        case Status::Ready:
            RawInput().Input(); 
            break;
        case Status::NotReady:
            break;
        default:
            break;
        }
        switch (HandleChanged) {
        case changeHandle::Passive:
            break;
        case changeHandle::Active:
            break;
        case changeHandle::Agressive:
            break;
        default:
            HandleChanged = changeHandle::Active;
            break;
        }
    }
};
int main()
{
    std::cout << "Hello World!\n";
}
