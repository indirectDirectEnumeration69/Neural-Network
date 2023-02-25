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
#include <functional>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <tuple>

template <typename T>
class ExternalManagingSystem {
public:
    virtual T ScrapeService() = 0;
    virtual T Connect() = 0;
    virtual T Webscraper() = 0;
    virtual T ConnectionCheck() = 0;
    virtual T ScrapeServiceChceck() = 0;
    virtual T serviceCheckStatus() = 0;

    class AdditionalManagement {
    public:
        AdditionalManagement() {
            hijack_manager = std::make_unique<HijackManager>();
        }
        virtual ~AdditionalManagement() = default;
        virtual void perform_hijack() {
            hijack_manager->HijackManager::hijack();
        }
    protected:
        class HijackManager {
        public:
            virtual ~HijackManager() = default;
            virtual void hijack() = 0;
        };
        std::unique_ptr<HijackManager> hijack_manager;
    };
    class NeuralNetworkManagement : public AdditionalManagement {
    public:
        virtual T train() = 0;
        virtual T predict() = 0;
    };
    class EncryptionSystem : public AdditionalManagement {
    public:
        virtual T encryptionSetup() = 0;
        virtual T encryptionCheck() = 0;
        virtual T encryptionCheckStatus() = 0;
        virtual T EncryptionChoice() = 0;
        virtual T CreateKey() = 0;
        virtual T KeyCheck() = 0;
        virtual T EncryptKey() = 0;
        virtual T CheckKeyEncryptionStatus() = 0;

        void perform_hijack() override {
            hijack_manager->HijackManager::hijack();
        }
    protected:
        class HijackManager : public AdditionalManagement::HijackManager {
        public:
            void hijack() override {
                ExternalManagingSystem<T> ConnectEMS;
                ConnectEMS.Connect();
            }
        };
        std::unique_ptr<HijackManager> hijack_manager;
    };
};

template <typename T>
class NeuralNetwork {
public:
    enum class NeuralMode
    {
        TrainingMode,
        LiveMode
    };
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
public:
    virtual std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Input() = 0;
    virtual T InputHandler() = 0;
    virtual T LayerGeneration() = 0;
    virtual T LayerOrganiser() = 0;
    virtual T CheckOrganisation() = 0;
    virtual T NeuronGeneration() = 0;
    virtual T NetworkScrape() = 0;
    virtual T TrainNetwork() = 0;
    virtual T SaveModel() = 0;
    class IntegratedThreadSystem {
    public:
        IntegratedThreadSystem() : stop_requested(false) {
            for (int i = 0; i < num_threads; ++i) {
                worker_threads.emplace_back(&IntegratedThreadSystem::worker_thread, this);
            }
        }
        ~IntegratedThreadSystem() {
            stop();
        }
        template<typename R, typename... Args>
        auto add_work(std::function<R(Args...)> work, Args... args) -> std::future<R> {
            auto promise = std::make_shared<std::promise<R>>();
            auto future = promise->get_future();
            std::function<void()> task = [promise, work, args...]() {
                promise->set_value(work(args...));
            };
            add_work_internal(task);
            return future;
        }
        auto add_work(std::function<void()> work) -> std::future<void> {
            auto promise = std::make_shared<std::promise<void>>();
            auto future = promise->get_future();
            std::function<void()> task = [promise, work]() {
                work();
                promise->set_value();
            };
            add_work_internal(task);
            return future;
        }
        void stop() {
            stop_requested.store(true, std::memory_order_relaxed);
            queue_cv.notify_all();
            for (auto& thread : worker_threads) {
                thread.join();
            }
        }
    private:
        static const int num_threads = 10;
        std::vector<std::thread> worker_threads;
        std::queue<std::function<void()>> work_queue;
        std::atomic<bool> stop_requested;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        void add_work_internal(std::function<void()> work) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                work_queue.push(work);
            }
            queue_cv.notify_one();
        }
        void worker_thread() {
            while (true) {
                std::function<void()> work;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [this]() { return !work_queue.empty() || stop_requested.load(std::memory_order_relaxed); });
                    if (stop_requested.load(std::memory_order_relaxed)) {
                        break;
                    }
                    work = std::move(work_queue.front());
                    work_queue.pop();
                }
                work();
            }
        }
    };
};
template<typename T>
class InputHandle : public NeuralNetwork<T> {
public:
    class RawInput : public NeuralNetwork<T> {
        std::vector<T>* x1;
        std::vector<T>* x2;
        std::vector<T>* x3;
        std::vector<T>* rawData;
    public:
        T InputHandler()override {
            return T();
        }
        T LayerGeneration()override {
            return T();
        }
        T LayerOrganiser()override {
            return T();
        }
        T CheckOrganisation()override {
            return T();
        }
        T NeuronGeneration()override {
            return T();
        }

        RawInput() :
            x1(new std::vector<T>()),
            x2(new std::vector<T>()),
            x3(new std::vector<T>()), rawData(new std::vector<T>()) {
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
                else {
                    x2->push_back(T());
                }
                if (i + 2 < rawData->size()) {
                    x3->push_back((*rawData)[i + 2]);
                }
                else {
                    x3->push_back(T());
                }
            }
        }
        ~RawInput() {
            delete x1;
            delete x2;
            delete x3;
            delete rawData;
        }  std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Input() override {
            return GetData();
        }

        std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> GetData() const {
            return std::make_tuple(*x1, *x2, *x3);
        }
    };template<typename T>
    class InputSystem : public RawInput {
    public:
        std::vector<T>* x1;
        std::vector<T>* x2;
        std::vector<T>* x3;
        InputSystem() :
            RawInput(),
            x1(new std::vector<T>()),
            x2(new std::vector<T>()),
            x3(new std::vector<T>())
        {
            std::tie(*x1, *x2, *x3) = RawInput::GetData();
        }
        std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> GetData() const {
            return std::make_tuple(*x1, *x2, *x3);
        }
        std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Input() override {
            return std::make_tuple(*x1, *x2, *x3);
        }
        ~InputSystem() {
            delete x1;
            delete x2;
            delete x3;
        }
        /*
        std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Input() override {
            RawInput rawInput{};
            std::vector<T> x1, x2, x3;
            std::tie(x1, x2, x3) = rawInput.GetData();
            return std::make_tuple(x1, x2, x3);
        }
        */
    };
    InputHandle(NeuralNetwork<T>::Status change, NeuralNetwork<T>::ChangeHandle handleChanged) {
        switch (change) {
        case NeuralNetwork<int>::Status::Ready: {
            InputSystem inputSystem{};
            std::vector<T> x1, x2, x3;
            std::tie(x1, x2, x3) = inputSystem.Input();
            break;
        }
        case NeuralNetwork<int>::Status::NotReady:
            break;
        default:
            break;
        }
        switch (handleChanged) {
        case NeuralNetwork<int>::ChangeHandle::Passive:
            break;
        case NeuralNetwork<int>::ChangeHandle::Active:
            break;
        case NeuralNetwork<int>::ChangeHandle::Agressive:
            break;
        default:
            handleChanged = NeuralNetwork<int>::ChangeHandle::Active;
            break;
        }
    }
};
template <typename T>
class Neuron : public NeuralNetwork<T> {
public:
    double weight;
    double bias;
    std::vector<int> inputIds;
    int neuronId;

    Neuron() : weight(0), bias(0), neuronId(0) {}

    T NeuronGeneration() override {
        std::vector<std::shared_ptr<Neuron<T>>> neurons1;
        std::vector<std::shared_ptr<Neuron<T>>> neurons2;
        std::vector<std::shared_ptr<Neuron<T>>> neurons3;
        std::vector<std::shared_ptr<Neuron<T>>> neurons4;

        for (int i = 0; i < 10; ++i) {
            neurons1.push_back(std::make_shared<Neuron<T>>());
            neurons2.push_back(std::make_shared<Neuron<T>>());
            neurons3.push_back(std::make_shared<Neuron<T>>());
            neurons4.push_back(std::make_shared<Neuron<T>>());
        }
        return std::make_tuple(neurons1, neurons2, neurons3, neurons4);
    }
    T CheckOrganisation() override {
        Neuron<T()> neuron;
        auto [neurons1, neurons2, neurons3, neurons4] = neuron.NeuronGeneration();
        auto checkNeuron = [](const std::shared_ptr<Neuron<double>>& neuron) {
            std::vector<std::shared_ptr<Neuron<T>>> neuronstoreBadVecs;
            std::vector<std::shared_ptr<Neuron<T>>> neuronstoreFineVecs;
            if (neuron->weight == 0.0 || neuron->bias == 0.0) {
                neuronstoreBadVecs.push_back(neuron);
            }
            else {
                neuronstoreFineVecs.push_back(neuron);
            }
            for (auto inputId : neuron->inputIds) {
                auto input = GetInitalInput(inputId);
                InputHandle<T> inputHandle(NeuralNetwork<T>::Status::Ready, NeuralNetwork<int>::ChangeHandle::Passive);
                auto [x1, x2, x3] = inputHandle.GetData();
                std::vector<T> dataRaw = *inputHandle.rawData;
                std::cout << "Input weight: " << input->weight << "\n";
                std::cout << "Input bias: " << input->bias << "\n";
            }

        };
        iterateNeurons(neurons1, checkNeuron);
        iterateNeurons(neurons2, checkNeuron);
        iterateNeurons(neurons3, checkNeuron);
        iterateNeurons(neurons4, checkNeuron);
    }
private:
    void iterateNeurons(const std::vector<std::shared_ptr<Neuron<T>>>& neurons, std::function<void(const std::shared_ptr<Neuron<T>>&)> checking) {
        for (auto& neuron : neurons) {
            checking(neuron);
        }
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
            std::vector<LayerNode> LayersNode;
            std::vector<NeuronNode> NeuronsNode;
        };
        NodeSystem() {
            for (int i = 0; i < 10; ++i) {
                LayersNode.push_back({ i, std::vector<int>() });
                for (int j = 0; j < 10; ++j) {
                    NeuronsNode.push_back({ j, std::vector<int>(), 0.0, 0.0 });
                    LayersNode.back().neuronIds.push_back(j);
                }
            }
        }
        std::vector<NodeTypes::LayerNode> LayersNode;
        std::vector<NodeTypes::NeuronNode> NeuronsNode;
    };
};
template <typename T>
class Layer : public NeuralNetwork<T> {
public:
    std::vector<std::shared_ptr<Neuron<T>>> neurons;
    std::vector<std::shared_ptr<Layer<T>>> LayersStorage;
    int layerId;
    int Layerthreshold;
    int Neuronthreshold;
    Layer() : layerId(0), Layerthreshold(10), Neuronthreshold(5) {
        LayersStorage.push_back(std::make_shared<Layer<T>>());
        LayersStorage.push_back(std::make_shared<Layer<T>>());
    }
    T InputHandler() {
        T Handle{};
        return Handle;
    }
    T LayerGeneration() {
        if (LayersStorage.empty()) {
            LayersStorage.push_back(std::make_shared<Layer<T>>());
        }
    }
    T LayerOrganiser() {
        for (const auto& layer : this->LayersStorage) {
            if (layer->neurons.size() < this->Layerthreshold) {
                for (int i = 0; i < this->Neuronthreshold - layer->neurons.size(); ++i) {
                    layer->neurons.push_back(std::make_shared<Neuron<T>>());
                }
            }
        }
    }
    T CheckOrganisation() {
        return T();
    }
};
template <typename T>
class TypeInputHandle : public InputHandle<T> {
public:
    using InputHandle<T>::InputHandle;
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

    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Input() override {
        return std::make_tuple(std::vector<T>(), std::vector<T>(), std::vector<T>());
    }

    T InputHandler() override {
        return T();
    }
    T NetworkScrape() override {
        return T();
    }
    T TrainNetwork() override {
        return T();
    }
    T SaveModel() override {
        return T();
    }
};

int main()
{
    TypeInputHandle<int>* inputHandle = new TypeInputHandle<int>(NeuralNetwork<int>::Status::Ready, NeuralNetwork<int>::ChangeHandle::Active);

    return 0;
}
/*
* Issue resolving type T for method return types so methods getdata and input are now unknown types<T>*
* calls constructor to get the input of rawdata class from the rawdata clss constructor function
* 
* 
* 
* Will be broken down into headers and modules for each class where i need to.
* 
* can do that in main , still compiles however needs to still be addressed ,simply just need to assert types for compile conditions.
* */

