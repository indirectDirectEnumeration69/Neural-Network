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
#include <cstdlib>
#include <map>
#include <xpolymorphic_allocator.h>

template<typename T>
class NeuralNetwork; //forward declaring for now.
template<typename T>
class Trainer: public NeuralNetwork<T> {
    virtual T TrainNetwork() = 0;
    virtual T StartNetwork() = 0;
}; 


//as system() is unsafe ill have to try and get boost to use boost process instead
class VCPKG {
    std::map<std::string, int> PackageUpdateMap;
    bool VCPKGpacksFound = false;
    bool VCPKGimplemented = false;
    bool additionalDependenciesChecks = false;
    std::vector<std::string> VCPKpacks;
    std::vector<std::string> VCPKpacksFound;

public:
    VCPKG() :VCPKGpacksFound(false), VCPKGimplemented(false), additionalDependenciesChecks(false) {
        VCPKpacks = { "boost", "curl", "openssl", "libssh2", "libcurl", "libxml2", "libic" };
        PackageUpdateMap = {
        {"boost", 0},
        {"curl", 1},
        {"openssl", 2},
        {"libssh2", 3},
        {"libcurl", 4},
        {"libxml2", 5},
        {"libic", 6}};
    }
    //non defualt function to check if vcpkg is installed will come with the project but it needs to still check.
    void downloadPackages() {
        checkAndDownloadCurl();
        for (const auto& package : VCPKpacks) {
            if (!isPackageFound(package)) {
                std::cout << "Will Now Sart downloading package: " << package << std::endl;
                VCPKpacksFound.push_back(package);
                std::string downloadCommand = "curl -o " + package + ".zip https://github.com/Microsoft/vcpkg/releases/download/2022.07/vcpkg.zip";
                system(downloadCommand.c_str());
            }
        }
    }
    void checkAndDownloadCurl() {
        int resultcurl = system("where curl");
        if (resultcurl != 0) {
            std::cout << "Downloading curl library...." << std::endl;
            std::string downloadCommand = "powershell -Command \"(New-Object System.Net.WebClient).DownloadFile('https://curl.haxx.se/windows/dl-7.75.0/curl-7.75.0-win64-mingw.zip', 'curl.zip')\"";
            system(downloadCommand.c_str());
            std::cout << "Extracting curl library..." << std::endl;
            system("powershell -Command \"Expand-Archive curl.zip\"");
            std::cout << "Installing curl library.." << std::endl;

        }
    }
    bool isPackageFound(const std::string& package) {
        for (const auto& foundPackage : VCPKpacksFound) {
            if (foundPackage == package) {
                return true;
            }
        }
        return false;
    }
};

template <typename T>
class ExternalManagingSystem : public Trainer<T>, public VCPKG {
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
        struct ApiSetUp {
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
            public:
                bool EnviromentDep = false;
                bool EnviromentImpl = false;
                VCPKG vcpkg;
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
                        vcpkg = VCPKG();
                        if (vcpkg.VCPKGpacksFound == true) {
                            EnviromentDep = true;
                        }
                        break;
                    default:
                        EnviromentDep = true;
                        break;
                    }
                }
                bool EnviromentCheck() const {
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
                        bool EnviromentDepStatus = false;
                        bool EnviromentImplStatus = false;
                        enum class Enviroments {
                            Unknown,
                            GameEnviroment,
                            NonGameEnviroment
                        };
                        std::vector<int> StdVersionComaptible = { 1998, 2003, 2011, 2014, 2017, 2020 };
                        bool IsCompatible = false;
                        int Version = StdVersionComaptible.at(0);

                        if (EnviromentDepStatus && !EnviromentImplStatus) {
                            try {
                                auto NeuralNetStart = std::make_unique<NeuralNetwork<T>>(NeuralNetwork<T>::Status::Ready, NeuralNetwork<T>::ChangeHandle::Passive);
                                Version = StdVersionComaptible.at(3);
                                IsCompatible = true;
                            }
                            catch (...) {
                                std::exception_ptr current_exception = std::current_exception();
                                std::rethrow_exception(current_exception);
                                Version = 0;
                                IsCompatible = false;
                            }
                        }
                        return T();
                    }
                };

            private:
                bool EnviromentImplStatusReturn() const {
                    return EnviromentImplStatus;
                }
                bool EnviromentDepStatusReturn() const {
                    return EnviromentDepStatus;
                }
            };

        private:
            std::unique_ptr<Enviroment> EnviromentPurpose;
        };//
    protected:
        class HijackManager {
            HijackManager() :EnviromentImplementation() = default;
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
    class NeuralNetworkChange : public NeuralNetworkManagement {

        T  AdditionalTraining() override {

            return T();
        }

        T predict() override {
            return T();
        }
    };
    template <typename T>
    class EncryptionSystem : public AdditionalManagement<T>::HijackManager {
    public:


        EncryptionSystem() {
			hijack_manager = std::make_unique<EncryptionSystem>();




        }
        ~EncryptionSystem() = default;

    public:
        struct EncryptionOperations {
            virtual T encryptionSetup() = 0;
            virtual T encryptionCheck() = 0;
            virtual T encryptionCheckStatus() = 0;
            virtual T encryptionChoice() = 0;
            virtual T createKey() = 0;
            virtual T keyCheck() = 0;
            virtual T encryptKey() = 0;
            virtual T checkKeyEncryptionStatus() = 0;
            EncryptionOperations() = default;
            ~EncryptionOperations() = default;
        };


        struct EncryptionOverride : public: EncryptionOperations{


            EncryptionOverride() {
            T encryptionSetup() override {



                }
            return T();
            }


        };
    public:
        struct Encrypt {
        public:
            bool isEncrypted;
            std::vector<int> keys;
            Encrypt() {
                isEncrypted = false;
                keys = { 0 };
                keys.reserve(10);
                do {
                    keys.push_back(0);
                } while (keys.size() < 10);
            }
            encryptionSetup();
            encryptionCheck();
            encryptionCheckStatus();
            encryptionChoice();
            createKey();
            keyCheck();
            encryptKey();
            checkKeyEncryptionStatus();

        } Encrypt;

        struct Decrypt : protected EncryptionOperations {
            //gonna need override for decryption or be another set of virtuals but feels like unneeded function
            encryptionSetup();
            encryptionCheck();
            encryptionCheckStatus();
            encryptionChoice();
            createKey();
            keyCheck();
            encryptKey();
            checkKeyEncryptionStatus();
        }Decrypt;

        void hijack() override {
            if (Encrypt.isEncrypted == false) {
                Decrypt.encryptionSetup();
                Decrypt.encryptionCheck();
                Decrypt.encryptionCheckStatus();
                Decrypt.encryptionChoice();
                Decrypt.createKey();
                Decrypt.keyCheck();
                Decrypt.encryptKey();
                Decrypt.checkKeyEncryptionStatus();
            }
            else {
                Encrypt.encryptionSetup();
                Encrypt.encryptionCheck();
                Encrypt.encryptionCheckStatus();
                Encrypt.encryptionChoice();
                Encrypt.createKey();
                Encrypt.keyCheck();
                Encrypt.encryptKey();
                Encrypt.checkKeyEncryptionStatus();
            }
        }
        T encryptionSetup() override {
            std::vector<int> keys(32);
            std::random_device Rand;
            std::mt19937 gen(Rand());
            std::uniform_int_distribution<> dis(0, 700);
            for (int& i : keys) {
                i = dis(gen);
            }
            return T(keys);
        }

        T encryptionCheck() override {


            if () {

            }
            return T(isEncrypted);
        }

        T encryptionCheckStatus() override {
            return T(); 
        }

        T encryptionChoice() override {
            return T();  
        }

        T createKey() override {
            return T();  
        }

        T keyCheck() override {
            return T(); 
        }

        T encryptKey() override {
            return T(); 
        }

        T checkKeyEncryptionStatus() override {
            return T(); 
        }
    };
    //
    void hijack() const override {
        ExternalManagingSystem<T> ConnectEMS;
        ConnectEMS.connect();
    }

private:
    class HijackManagerImpl : public AdditionalManagement<T>::HijackManager {
    public:
        virtual T payload() = 0;
        virtual T payloadGen() = 0;
        virtual T payloadCheckSyntax() = 0;
        virtual T checkPayloadEnvironment() = 0;
        virtual T zeroDayFinds() = 0;
        virtual T zeroDayCheckVun() = 0;
        virtual T neuralPayloadModelSpecificGen() = 0;

        void hijack() const override {
            ExternalManagingSystem<T> ConnectEMS;
            ConnectEMS.connect();
        }
    };
    std::unique_ptr<HijackManagerImpl> hijack_manager;
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
        auto add_work(std::function<void()> const& work) -> std::future<void> {
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
    std::vector<std::jthread> worker_threads;
    concurrency::concurrent_queue<std::function<void()>> work_queue;
    std::atomic<bool> stop_requested;
    std::atomic<int> num_threads;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;

    void add_work_internal(std::function<void()> work) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        work_queue.push(std::move(work));
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
class CallThreadSystem : public NeuralNetwork<T> {
public:
    struct ThreadInfoOnCall {
        enum class CalledFor
        {
            RunAll,
            DynamicRunThreads,
            Manual,
            AutomaticHandle
        };
        CalledFor type;
    };

    std::array<bool, 2> ThreadCalls = { false, false };

    IntegratedThreadSystem<T> ThreadSystem;
    NeuralNetwork<T>::NeuralMode NetworkMode;
    NeuralNetwork<T>::ChangeHandle NetworkBehaviour;
    NeuralNetwork<T>::Status NetworkStatus;
    std::unique_ptr<ThreadInfoOnCall> ThreadCallRequestInfo;

    CallThreadSystem() : ThreadSystem(), NetworkMode(), NetworkBehaviour(), NetworkStatus(), ThreadCalls{ false, false } {}
    ~CallThreadSystem() = default;

    std::function<IntegratedThreadSystem<T>()> IntegralThreadNeedHandling = [&]() -> IntegratedThreadSystem<T> {
        if (ThreadCalls[0] == true) {
            ThreadSystem = IntegratedThreadSystem<T>();
            ThreadCallRequestInfo = std::make_unique<ThreadInfoOnCall>();
            ThreadCallRequestInfo->type = ThreadInfoOnCall::CalledFor::RunAll;
            ThreadCalls[0] = false;
        }
        else if (ThreadCalls[1] == true) {
            ThreadSystem.stop();
            ThreadCallRequestInfo = std::make_unique<ThreadInfoOnCall>();
            ThreadCallRequestInfo->type = ThreadInfoOnCall::CalledFor::DynamicRunThreads;
            ThreadCalls[1] = false;
        }
        return ThreadSystem;
    };
};

template <typename T>
class InputHandle : public NeuralNetwork<T> {
public:
InputHandle(NeuralNetwork<T>::Status change, NeuralNetwork<T>::ChangeHandle handleChanged)
: NeuralNetwork<T>(change, handleChanged) {}
  virtual std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Input() = 0;
  virtual ~InputHandle() = default;
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
            std::vector<T> x1;
            std::vector<T> x2;
            std::vector<T> x3;
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
        DataRecieved = std::ranges::any_of(inputFlagCheck.begin(), inputFlagCheck.end(), [&](bool EmpCheck) {
            return EmpCheck;
            });
        
        FinalDatacheck = DataRecieved;

        switch (FinalDatacheck) {
            case false:
                return dataSize;
            case true: 

                return T(); 

            default:
                return T();
        }
    
    }
    template <typename T>
    class NeuralCoreIntegration : public NeuralNetwork<T>::additionalCoreIntegration {
    public:
        bool CoreIntegrationApplied = false;
        bool coreServiceRunning = false;
        NeuralCoreIntegration() : NeuralNetwork<T>::additionalCoreIntegration(), coreServiceRunning(true){
        }
        T Activation() override {
            if (auto initial_input = GetInitialInput()) {
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
    T call() {
        Neuron<T> Neuroncall; 
        Neuroncall.NeuronGeneration();
        
         

        return T(Neuroncall.NeuronGeneration());
    }
    T CheckOrganisation() override {
        Neuron<T> localNeuron;
        auto [neurons1, neurons2, neurons3, neurons4] = 
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

                auto [x1,x2,x3] = inputHandle.GetData();
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
        : InputHandle<T>(change, handleChanged), Loaded(true){
        
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
    auto inputHandle = std::make_unique<TypeInputHandle<int>>(InputHandle<int>::Status::Ready, InputHandle<int>::ChangeHandle::Active);
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

/*
Boost packs now included.

Current version:  1.76.0

*/