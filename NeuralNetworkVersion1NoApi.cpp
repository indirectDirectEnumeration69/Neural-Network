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
#include <type_traits>
#include <variant>


template<typename T>
class NeuralNetwork; //forward declaring for now.
template<typename T>
class Trainer:NeuralNetwork<T> {
    virtual T TrainNetwork() = 0;
    virtual T StartNetwork() = 0;
};


template <typename T>
class ExternalManagingSystem:Trainer<T>{
public:
    virtual T SetApi() = 0;
    virtual T API() = 0;
    virtual T PLantAPi() = 0;
    virtual T PlantAPiCheck() = 0;
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
        struct WebService {
            WebService() {

            }
            T ScrapeService() override {

                return T();
            }
            T Connect() override {
                //curl and boost.
                return T();
            }
            T Webscraper() override {

                return T();
            }
            T ConnectionCheck() override {

                return T();
            }
            T ScrapeServiceChceck()  override {

                return T();
            }
            T serviceCheckStatus() override {

                return T();
            }
        };
        struct ApiSetUp{
            ApiSetUp() {}
            T SetApi() override {
                struct ApiInfomation {
                    std::unique_ptr<EncryptionSystem> EncryptionKey;
                    float ApiNumber;
                    std::string ApiName;
                    ApiInfomation() : EncryptionKey(std::make_unique<EncryptionSystem>()), ApiNumber(0.0), ApiName("") {}
                };
                float num = 0;
                std::function<ApiInfomation()> ApiSetUp = [&num]() {
                    ApiInfomation ApiInfo;
                    ApiInfo.EncryptionKey = std::make_unique<EncryptionSystem>();
                    ApiInfo.ApiNumber = num++;
                    ApiInfo.ApiName = "NewNetworkApi";
                    return ApiInfo;
                };
                return ApiSetUp();
            }
            
            T API() override
            {
                T ApiConnection = WebService::Connect();
                T ApiSetup = SetApi();
                std::function<ApiInfomation()> ApiKeyGeneration = [&ApiSetup]() {

                    if (ApiConnection == false) {




                        ApiConnection = WebService::ConnectionCheck();

                        if (ApiConnection == false) {




                            return ApiConnection;
                        }
                        return ApiConnection;
                  }



                };
            }
            T PLantAPi() override
            {

            }
            T PlantAPiCheck() override
            {

            }
        };
        class EnviromentImpleMentation {
            public:

            EnviromentImplementation() {

            }
            enum class ImplementationStatus
            {
                Successful,
                Failed,
                Waiting,
                Loading,
                CantFindEnv,
                FoundEnviroment
            };
            enum class EnviromentDependencies {
                FoundEnviromentDependencies,
                CouldntFindEnviromentDependencies,
                WaitingToFindDependencies
            };
            struct EnviromenDynamicActions {
                virtual T FindEnviroment() = 0;
            };
            class Enviroment {
                bool EnviromentDep = false;
                bool EnviromentImpl = false;
                Enviroment() {
                    auto EnvImpstatus = ImplementationStatus::Waiting;
                    auto Depstatus = EnviromentDependencies::WaitingToFindDependencies;
                    switch (EnvImpstatus) {
                    case ImplementationStatus::Waiting:
                        EnviromentImpl = false;
                        break;
                    default:
                        EnviromentImpl = true;
                        break;
                    }

                    switch (Depstatus) {
                    case EnviromentDependencies::WaitingToFindDependencies:
                        EnviromentDep = false;
                        break;
                    default:
                        EnviromentDep = true;
                        break;
                    }
                }
                bool EnviromentCheck() {
                    if (EnviromentImpl == true && EnviromentDep == true) {
                        return true;
                    }
                    else {
                        return false;
                    }
                }
                class DynamicActions : public EnviromentImplementation::EnviromenDynamicActions {
                public:
                    T FindEnviroment() override {
                        EnviromentDep = false;
                        EnviromentImpl = false;
                        enum Enviroments {
                            Unknown,
                            GameEnviroment,
                            NonGameEnviroment
                        };
                        auto EnviromentGame = Enviroments::GameEnviroment;
                        auto EnviromentNon = Enviroments::NonGameEnviroment;
                        int Version = 0000;
                        std::vector<int> StdVersionComaptible = { 1998, 2003, 2011, 2014, 2017, 2020 };
                        bool CurrentlyCompatible = false;
                        std::string EngineName = "";
                        int Version = StdVersionComaptible.at(0);
                        if (EnviromentDep && EnviromentImpl == false) {
                            auto EnviromentNow = Enviroments::Unknown;
                        }
                        if (EnviromentNow == Enviroments::GameEnviroment) {
                            try {
                                std::function<void()> GetGameEnviroment = [&]() {
                                    NeuralNetwork<T>* NeuralNetStart = new NeuralNetwork<T>(NeuralNetwork<T>::Status::Ready, NeuralNetwork<T>::ChangeHandle::Passive);





                                    Version = StdVersionComaptible.at(3);
                                };
                            }
                            catch (...) {
                                std::exception_ptr current_exception = std::current_exception();
                                std::rethrow_exception(current_exception);
                                Version = 0000;
                            }
                        }
					return T();
                    }
                };
            private:
                bool EnviromentImplStatusReturn() {
                    return EnviromentImpl;
                }
                bool EnviromentDepStatusReturn() {
                    return EnviromentDep;
                }
            };

        private:
			std::unique_ptr<Enviroment> EnviromentPurpose;
        };

    protected:
        class HijackManager {
            HijackManager():EnviromentImplementation() {

            }
        public:
            virtual ~HijackManager() = default;
            virtual void hijack() = 0;
        };
        std::unique_ptr<HijackManager> hijack_manager;
        std::unique_ptr<EnviromentImplementation> EnviromentDynamicPurpose;
    };

    class NeuralNetworkManagement : public AdditionalManagement {
    public: 
        virtual T AdditionalTraining() = 0;
        virtual T predict() = 0;
    };
    class NeuralNetworkChange: public NeuralNetworkManagement{
        
        T  AdditionalTraining() override{

            return T();
        }
        
        T predict() override {
			return T();
		}
    };

    class EncryptionSystem : public AdditionalManagement::HijackManager {
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
            virtual T payload() = 0; //crucial for using system gpus to acquire additional performance for the network to use.
            virtual T payloadGen() = 0;
            virtual T payloadCheckSyntax() = 0;
            virtual T checkPayloadEnvironment() = 0;
            virtual T ZeroDayFinds() = 0;
            virtual T ZeroDayCheckVun() = 0;
            virtual T NeuralPayloadModelSpecificGen() = 0;
        public:
            void hijack() override {
                ExternalManagingSystem<T> ConnectEMS;
                ConnectEMS.Connect();
            }
            HijackManager() {

            }
        };
        std::unique_ptr<EncryptionSystem::HijackManager> hijack_manager;
    };
};
//
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

private:
    Status change;
    ChangeHandle handleChanged;

public:
    NeuralNetwork(Status change, ChangeHandle handleChanged) : change(change), handleChanged(handleChanged) {}

    virtual std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Input() = 0;
    virtual T InputHandler() = 0;
    virtual T LayerGeneration() = 0;
    virtual T LayerOrganiser() = 0;
    virtual T CheckOrganisation() = 0;
    virtual T NeuronGeneration() = 0;
    virtual T NetworkScrape() = 0;
    virtual T TrainNetwork() = 0;
    virtual T SaveModel() = 0;


    struct ThreadDynamicOptimisationSystem {
        bool Optimise = false;
        virtual T ThreadDynamicOptimisation() = 0;

    };
    struct additionalCoreIntegration {
        virtual T NetworkFeedForward() = 0;
        virtual T NetworkBackPropogation() = 0;
        virtual T TotalWeights() = 0;
        virtual T TotalBiases() = 0;
        virtual T SetTrainingLoop() = 0;
        virtual T PerformanceEvalNeuronTotal() = 0;
        virtual T NeuralTrainingLoopSet() = 0;
    };
}; // 

template <typename T>
class IntegratedThreadSystem :NeuralNetwork<T> {
    NeuralNetwork<T>::ThreadDynamicOptimisationSystem::Optimise ThreadOptimisation = false;
public:
    IntegratedThreadSystem() : stop_requested(false), num_threads(std::thread::hardware_concurrency()) {
        worker_threads.reserve(num_threads);
        if (ThreadOptimisation == false) {
            for (int i = 0; i < num_threads; ++i) {
                worker_threads.emplace_back(&IntegratedThreadSystem::worker_thread, this);
            }
        }
        else if (ThreadOptimisation) {
            for (int i = 0; i < worker_threads.size(); ++i) {
                ThreadDynamicOptimisation();
            }
        }
    }~IntegratedThreadSystem() {
            stop();
        }
    template<typename R, typename... Args>
    auto add_work(std::function<R(Args...)> work, Args... args) -> std::future<R> {
        std::variant<std::promise<R>, std::promise<void>> promise;
        if constexpr (std::is_same_v<R, void>) {
            promise = std::promise<void>();
        }
        else {
            promise = std::promise<R>();
        }
        auto future = std::visit([](auto& p) -> std::future<R> { return p.get_future(); }, promise);

        std::function<void()> task = [promise = std::move(promise), work = std::move(work), args...]() mutable {
            try {
                auto result = work(args...);
                std::visit([&](auto& p) { using T = decltype(p); if constexpr (!std::is_same_v<T, std::promise<void>>) p.set_value(result); }, promise);
            }
            catch (...) {
                std::visit([&](auto& p) { p.set_exception(std::current_exception()); }, promise);
            }
        };
        add_work_internal(task);
        return future;
    }
        auto add_work(std::function<void()> work) -> std::future<void> {
            auto promise = std::make_shared<std::promise<void>>();
            auto future = promise->get_future();

            std::function<void()> task = [promise, work]() {
                try {
                    work();
                    promise->set_value();
                }
                catch (...) {
                    promise->set_exception(std::current_exception());
                }
            };

            add_work_internal(task);
            return future;
        }

        void stop() {
            stop_requested = true;
            queue_cv.notify_all();
            for (auto& thread : worker_threads) {
                thread.join();
            }
        }
protected:
    T ThreadDynamicOptimisation() override {
        return T();
    }
private:
    std::vector<std::thread> worker_threads;
    concurrency::concurrent_queue<std::function<void()>> work_queue;
    std::atomic<bool> stop_requested;
    std::atomic<int> num_threads;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    void add_work_internal(std::function<void()> work) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        work_queue.push(work);
        lock.unlock();
        queue_cv.notify_one();
    }
    void worker_thread() {
        while (true) {
            std::optional<std::function<void()>> work;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this]() { return stop_requested.load(std::memory_order_relaxed) || work_queue.try_pop(work); });
                if (stop_requested.load(std::memory_order_relaxed)) {
                    break;
                }
            }
            if (work.has_value()) {
                work.value()();
            }
        }
    }
};
template<typename T>
class CallThreadSystem:NeuralNetwork<T>{
public:

    struct ThreadInfoOnCall {

        ThreadInfoOnCall() : () {};
    };

    bool* ThreadCalls = new bool[2];

    IntegratedThreadSystem<T> ThreadSystem;
    NeuralNetwork<T>::NeuralMode<T> NetworkMode;
    NeuralNetwork<T>::ChangeHandle<T> NetworkBehaviour;
    NeuralNetwork<T>::Status<T> NetworkStatus;



    CallThreadSystem() : ThreadSystem(), NetworkMode(), NetworkBehaviour(), NetworkStatus(), ThreadCalls{ false, false } {};
    ~CallThreadSystem() {
		delete[] ThreadCalls;
	};
    std::function<IntegratedThreadSystem<T>()> IntegralThreadNeedHandling = [&]() -> IntegratedThreadSystem<T>{

       




        return ThreadSystem;
    };
};


template <typename T>
class InputHandle : public NeuralNetwork<T> {
public:
InputHandle(NeuralNetwork<T>::Status change, NeuralNetwork<T>::ChangeHandle handleChanged)
: NeuralNetwork<T>(change, handleChanged) {}
  virtual std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Input() = 0;

    template <typename T>
    class RawInput {
        std::vector<std::variant<int, double, float>> rawData;
    public:
        T InputHandler() { return T(); }
        T LayerGeneration() { return T(); }
        T LayerOrganiser() { return T(); }
        T CheckOrganisation() { return T(); }
        T NeuronGeneration() { return T(); }
        RawInput() {
            std::string UserInput;
            std::vector<std::variant<int, double, float>> DataRaw;

            while (std::getline(std::cin, UserInput)) {
                std::istringstream inputStream(UserInput);
                std::string data;

                if (inputStream >> data) {
                    try {
                        DataRaw.push_back(std::stoi(data));
                    }
                    catch (const std::invalid_argument&) {
                        try {
                            DataRaw.push_back(std::stod(data));
                        }
                        catch (const std::invalid_argument&) {
                            try {
                                DataRaw.push_back(std::stof(data));
                            }
                            catch (const std::invalid_argument&) {
                               
                            }
                        }
                    }
                }
                else {
                    if (inputStream.eof()) {
                        break;
                    }
                    inputStream.clear();
                }
            }
        }
        std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Input() {
            return GetData();
        }
        std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> GetData() const {
            std::vector<T> x1, x2, x3;
            const int size = rawData.size();
            for (int i = 0; i < size; i += 3) {
                const auto x1_value = std::get_if<T>(&rawData[i]);
                x1.push_back(x1_value ? *x1_value : T());

                if (i + 1 < size) {
                    const auto x2_value = std::get_if<T>(&rawData[i + 1]);
                    x2.push_back(x2_value ? *x2_value : T());
                }
                else {
                    x2.push_back(T());
                }

                if (i + 2 < size) {
                    const auto x3_value = std::get_if<T>(&rawData[i + 2]);
                    x3.push_back(x3_value ? *x3_value : T());
                }
                else {
                    x3.push_back(T());
                }
            }
            return std::make_tuple(x1, x2, x3);
        }
    };
    template <typename T>
    class InputSystem : public RawInput<T> {
    public:
        std::vector<T> x1;
        std::vector<T> x2;
        std::vector<T> x3;
        InputSystem() :
            RawInput<T>(),
            x1(),
            x2(),
            x3()
        {
            std::tie(x1, x2, x3) = RawInput<T>::GetData();
        }

        std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> GetData() const {
            return std::make_tuple(x1, x2, x3);
        }

        std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Input() override {
            return std::make_tuple(x1, x2, x3);
        }
    };
};
template <typename T>
class Neuron : public NeuralNetwork<T> {
public:
    double weight;
    double bias;
    std::vector<int> inputIds;
    int neuronId;
    InputHandle::RawInput<T>* inputHand;
    bool DataRecieved = false;
    bool FinalDatacheck = false;
    std::vector<std::pair<T, T>>* NeuronData = nullptr;
    struct NeuronSpecific {
        virtual T Activation() = 0;
        virtual T Loss() = 0;
        virtual T Optimisation() = 0;
        virtual T FeedForwards() = 0;
        virtual T BackPropogation() = 0;
        virtual T UpdateBiases() = 0;
        virtual T UpdateWeight() = 0;
        virtual T NeuronPerformanceEval() = 0;
    };
    Neuron() : weight(0), bias(0), neuronId(0), inputHand(nullptr) {
    }
    T GetInitialInput() {
        auto GotDataInput = inputHand->GetData();
        std::vector<T>& x1 = std::get<0>(gotDataInput);
        std::vector<T>& x2 = std::get<1>(gotDataInput);
        std::vector<T>& x3 = std::get<2>(gotDataInput);
        std::size_t dataSize = std::max({ x1.size(), x2.size(), x3.size() });
        std::vector<bool> inputFlagCheck(dataSize);
        inputIds.resize(dataSize);
        for (std::size_t i = 0; i < dataSize; ++i) {
            inputIds[i] = i;
            inputFlagCheck[i] = (i < x1.size() && i < x2.size() && i < x3.size());
        }
        DataRecieved = std::any_of(inputFlagCheck.begin(), inputFlagCheck.end(), [&](bool EmpCheck) {
            return EmpCheck;
            });
        
        FinalDatacheck = DataRecieved;

        switch (FinalDatacheck) {
            case false:

                return dataSize;

            case true: 






                return T();
        }
    
    }
    class NeuralCoreIntegration : public NeuralNetwork<T>::additionalCoreIntegration {
    public:
        bool CoreIntegrationApplied = false;
        bool coreServiceRunning = false;
        NeuralCoreIntegration() : NeuralNetwork<T>::additionalCoreIntegration() {
            coreServiceRunning = true; 
        }
        T Activation() override {
            auto initial_input = GetInitialInput();
            if (initial_input) {
                return std::max(T(0), weight * *initial_input + bias);
            }
            else {
                
            }
            return std::max(T(0), weight * GetInitialInput() + bias);
        }
        T Loss() override {

        }
        T Optimisation() override {

        }
        T FeedForwards() override {

        }
        T BackPropogation() override {
            return std::tanh(weight * GetInitialInput() + bias);
        }
        T UpdateBiases() override {

        }
        T UpdateWeights() override {
        }
        T PerformanceEval() override {
        }

        T NetworkFeedForward() = 0;
        T NetworkBackPropogation() = 0;
        T TotalWeights() = 0;
        T TotalBiases() = 0;
        T SetTrainingLoop() = 0;
        T PerformanceEvalNeuronTotal() = 0;
        T NeuralTrainingLoopSet() = 0;
    };
    T NeuronGeneration(T inputId) override {
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
        Neuron<T> neuron;
        auto [neurons1, neurons2, neurons3, neurons4] = neuron.NeuronGeneration();
        auto checkNeuron = [](const std::shared_ptr<Neuron<T>>& neuron) {
            std::vector<std::shared_ptr<Neuron<T>>> neuronstoreBadVecs;
            std::vector<std::shared_ptr<Neuron<T>>> neuronstoreFineVecs;
            if (neuron->weight == 0.0 || neuron->bias == 0.0) {
                neuronstoreBadVecs.push_back(neuron);
            }
            else {
                neuronstoreFineVecs.push_back(neuron);
            }
             for (T inputId : neuron->inputIds) {
               auto input = GetInitalInput(inputId);
               InputHandle<T> inputHandle(NeuralNetwork<T>::Status::Ready, NeuralNetwork<double>::ChangeHandle::Passive);
               
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
        std::vector<typename NodeTypes::LayerNode> LayersNode;
        std::vector<typename NodeTypes::NeuronNode> NeuronsNode;
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
    T LayerOrganiser() { //fixing type def issues
        for (const auto& layer : this->LayersStorage) {
            if (layer-><T>neurons.size() < this->Layerthreshold) {
                for (int i = 0; i < this->Neuronthreshold - layer-><T>neurons.size(); ++i) {
                    layer-><T>neurons.push_back(std::make_shared<Neuron<T>>());
                }
            }
        }
    }
    T CheckOrganisation() {
        int NeuronsTotal = 0;
        for (const auto& layer : this->LayersStorage) {
            if (layer-><T>neurons.size() != this->Neuronthreshold) {
                return T(false);
            }
            NeuronsTotal += layer-><T>neurons.size();
        }
        if (NeuronsTotal != this->Layerthreshold * this->Neuronthreshold) {
            return T(false);
        }
        return T(true);
    }
};
template <typename T>
class TypeInputHandle : public InputHandle<T> {
    using DataVariant = std::variant<int, double, float>;
    bool Loaded = false;
    std::unique_ptr<DataVariant> TypePoint = nullptr;
public:
    TypeInputHandle(InputHandle<T>::Status change, InputHandle<T>::ChangeHandle handleChanged)
        : InputHandle<T>(change, handleChanged) {
        Loaded = true;
        TypePoint = std::make_unique<DataVariant>(DataVariant{});
    }
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

int main() {
    TypeInputHandle<int>* inputHandle = new TypeInputHandle<int>(InputHandle<int>::Status::Ready, InputHandle<int>::ChangeHandle::Active);
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
* 
* 
* Structure maybe abit more optimisation and ill need some additional implemntation of boost and unreal engine version checking but ill have to check based on 2 options 1. i use current std 20 to check 
* but if the version is not compatible with the newer version of std it will fail.
* , this can be fixed with std14 compatible functionality hence why ive went for a flexible dynamic network , however functionality to rewrite the classes and replace them with c++ 14 compatible functions will be done.
* Or in Engine source code compatibility fixing for std::20 can be done dynamically possibly .
* 
* 
* more errors due to additional implementation , also i need to use the vpkg packages for the boost libs for the boost network activity, ill just do it in terminal for the gitclone.
* */

