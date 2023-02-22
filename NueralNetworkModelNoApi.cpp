// NueralNetworkModelNoApi.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
/*
The network class contains a set of layers that process and categorize the data.
Each layer contains a set of neurons and a set of inputs.
Each nueron contains a set of weights and a bias.
The weights determine the influence of each input on the neurons output.
The bias shifts the activation function of the neuron and represents the importance of the nueron in terms of the data being processed by the network.
The inputs are the data that is being processed by the network.
-layers contains the hidden layers and the output layer.
-each layer has nuerons which recieve inputs from a previous layer and use the input data to then calculate the y output value then
pass the output value to the next layers neurons.
//
each neuron takes in x1,x2 and x3 which contain w1 ,w2 ,w3 passed into the neuron and then the neuron calculates the output y.

Types of input data:
inputs: z = w^tx = w1x1+w2x2+w3x3
inputs: z = w^tx = w1x1+w2x2+w3x3+bias

Types of activation functions:
Activation: F(z) = 1/(1+e^-z) // inorder to map number values to a probability value between 0 and 1.
							// 0 and 1 then determines the instances likelyhood of existing in a class between the s curve of 0 to 1. 
							//0 is represented as a class and 1 the other class for the neuron to determine the output.
							//obviosuly the classification boundary for dertmination is 0.5 for classing the data input.
Activation: F(z) = max(0,z)

Types of output data:
output y = F(z) = 1/(1+e^-z)
output:  y = F(z)

Types of loss functions:
loss: L(y,y') = (y-y')^2
loss: L(y,y') = max(0,1-y*y')

The loss function bases the error of the network on the difference between the expected output and the actual output.
this allows for the network to be trained to improve accuracy then produce a gradient to mark the performance in real time.
//
-inputs are taken in as a vector of doubles which is passed to the first layer of the network.
Vectors contain the data for the network but arnt directly involved with the network.
*/
/*
Regression issues:
issues in trying to predict a value that is not a classificatio based on a constant value term and independant predictor variables to
maintain predicted output accuracy.
*/


/*
Layers:
Feeding forwards through the network:
Input layer - > [ Hidden layer 1 - > Hidden layer 2 -> Hidden Layer 3 ] - > Output layer.

hidden 1 and 2 and 3 can be classified as the black box within a algorithm. input -> [blackbox] -> output 


Backpropagation through the network:
Output layer - > [ Hidden layer 3 - > Hidden layer 2 -> Hidden Layer 1 ] - > Input layer.

feed data in then pass then using the output of the network to calculate the error of the network.
then pass the error back through the network to calculate the gradient of the network.
then use the gradient to update the weights and biases of the network.
then repeat the process until the network is accurate enough to be used for predictions.



   //optimisation note on predictors:
//reptition esssentially allows for calibration of each nueron within each
//layer for minimisation of inaccuracies between constant and predicator variables.
*/

//basic needs for this:
/*
Networks class.
Layers class.
Neuron class.
*/

/*
class Neuron
{
public:
Neuron() {}
	Neuron(int Inputs) {}
	void setWeights(std::vector<double> weights) {}
	void setBias(double bias) {}
	double getOutput(std::vector<double> inputs) { return 0; }
	std::vector<double> getWeights() { 
	return std::vector<double>(); 
	}
	double getBias() {
	return 0;
	}
	void print() {}
};
class Layer
{
public:
	Layer() {}
	Layer(int Neurons, int Inputs) {}
	void setNeurons(std::vector<Neuron> neurons) {}
	std::vector<Neuron> getNeurons() { 
	return std::vector<Neuron>(); 
	}
	void print() {}
};
class Network
{
public:
	Network() {}
	Network(std::vector<int> layers) {}
	void setLayers(std::vector<Layer> layers) {}
	std::vector<Layer> getLayers() { 
	return std::vector<Layer>(); 
	}
	void print() {}
};

*/
//the layers are for the hidden layers and the output layer
// a layer is a vector of neurons 
//and so a neuron is a vector of weights and a bias for determining relevance and importance.
// the input layer is a vector of inputs and is not included within any of the layers vectors which contains neurons for other vectors. 
//The input layer is a vector of inputs which are not designated to a neuron set.


// Input layer is just storage for the vectors of inputs getting passed to the first hidden layer in the network.
/*
								 ___________Hidden_________Layer_____One____________--------- Hidden layer 2 ----------| Hidden layer 3 ---------------------
								|		   _______					 neuron 1	    |								   |
                              	|	 x1 ->|Neurons|_		Neuron 3 neuron 2	    |		<Home in the data>												_____OUTPUT____
input variables vectors[x1,x2,x3] => x2 ->|		  |_ = = >	Neuron 2 neuron 3		=> 		neurons 1-6  -> 1-3 neurons  - neurons 1-9 -> 1-27 -> 1-81 -> -> ->	1 neuron -> response. 		
								|    x3-> |_______| (w^T x)	Neuron 1 neuron 4       |																		_______________
								|	     				             neurons 5-9    |
								|___________________________________________________|__________________________________|_____________________________________
*/
#include <iostream>
#include <vector>
#include <math.h>
#include <thread>
#include <mutex>
#include <future>
#include <concurrent_unordered_set.h>
#include <sstream>

template <typename T>
class NetWorkInterface {
public:
    virtual void inputlayer() = 0;
    virtual void hiddenlayer() = 0;
    virtual void outputlayer() = 0;
    virtual std::string RawInput() = 0;
    virtual void setLayers() = 0;
    virtual void setNeurons() = 0;

    class RawInputX {
        std::string raw;
        std::vector<T> data;
        T value{};
    public:
        std::vector<std::pair<T, T>> RawInput() {
            std::cin >> raw;
            std::istringstream stringstreamer(raw);
            while (stringstreamer >> value) {
                data.push_back(value);
            }
            std::vector<std::pair<T, T>> relationCoincidental;
            for (int i = 0; i < data.size(); i += 2) {
                if (i + 1 < data.size()) {
                    relationCoincidental.push_back(std::make_pair(data[i], data[i + 1]));
                }
            }
            return relationCoincidental;
        }
    };
    class InputLayer : public RawInputX {
    public:
        InputLayer() {
            auto pairs = RawInputX::RawInput();
            for (auto& relation : pairs) {
                x1.push_back(relation.first);
                x2.push_back(relation.second);
                if (relation != pairs.back()) {
                    auto& next_relation = *(++ & relation);
                    x3.push_back(next_relation.first);
                }
                else {
                    x3.push_back(0);
                }
            }
        }
    protected:
        std::vector<T> x1;
        std::vector<T> x2;
        std::vector<T> x3;
    };
    class Neuron {
    private:
        std::vector<T> inputs = {};
        double bias = 0;
        double weight = 0;
        std::vector<double>* weights = nullptr;
        std::vector<double> outputs;

        struct inputData {
            inputData(const std::vector<T>& inputs) {
                using inputT = typename std::conditional<std::is_same<T, double>::value, double, float>::type;
                std::vector<inputT> sortedInputs;
                if (!inputs.empty()) {
                    if (std::is_same<T, double>::value) { //categorise inputs by there types.
                        sortedInputs = std::vector<inputT>(inputs.begin(), inputs.end());
                    }
                    else {
                        sortedInputs = std::vector<inputT>(inputs.begin(), inputs.end());
                    }
                }
            }
        };

    public:
        Neuron() {
            weights = new std::vector<double>();
        };

        Neuron(std::vector<double>* weightpointer) {
            if (weights != weightpointer) {
                weights = new std::vector<double>();
            }
        }
        void setWeights(std::vector<double>& weights) {
            this->weights = &weights;
        }

        void setBias(double bias) {
            this->bias = bias;
        }

        double ActivationOutput(std::vector<double> inputs) { return 0; }

        std::vector<double> getWeights() {
            return std::vector<double>();
        }
    };

    class Layers {
    private:
        std::vector<int>layerTotal;
        int NeuronsCurrentTotal;
        int CurrentInputsTotal;

    public:
        Layers() {
        inputlayer();
        hiddenlayer();
        outputlayer();
        }
        Layers(int NeuronsCurrentTotal, int CurrentTotalInputs) : NeuronsCurrentTotal(NeuronsCurrentTotal), CurrentInputsTotal(CurrentInputsTotal) {
            inputlayer();
            hiddenlayer();
            outputlayer();
        }

        void setNeurons(std::vector<Neuron> neurons) {}
        std::vector<Neuron> getNeurons() {
            return std::vector<Neuron>();
        }

        void LayerInfo() {}
    };

    class Network {
    public:
        Network() {}
        Network(std::vector<int> layers) {
        
        }

        void setLayers(std::vector<Layers> layers) {}
        std::vector<Layers> getLayers() { return std::vector<Layers>(); }

        void print() {}
    };
private: 
    virtual void feedforward() = 0;
    virtual void backpropagation() = 0;
    virtual void train() = 0;
    virtual void predict() = 0;
};


template<typename T>
class ThreadManagement : public NetWorkInterface<T> {
public:
    virtual std::vector<std::future<void>> getFutures() = 0;
    virtual void setFutures(std::vector<std::future<void>> futures) = 0;

    void inputlayer() override {

    }
    void hiddenlayer() override {

    }
    void outputlayer() override {
    }
    void setLayers(std::vector<typename NetWorkInterface<T>::Layers> Layers) override {
        typename NetWorkInterface<T>::Network network{};
        network.setLayers(Layers);
    }
    std::vector<typename NetWorkInterface<T>::Layers> getLayers() override {
        return std::vector<typename NetWorkInterface<T>::Layers>();
    }
    void RunEverything() {
        std::vector<std::future<void>> futures;
        futures.push_back(std::async(std::launch::async, &ThreadManagement::inputlayer, this));
        futures.push_back(std::async(std::launch::async, &ThreadManagement::hiddenlayer, this));
        futures.push_back(std::async(std::launch::async, &ThreadManagement::outputlayer, this));
        futures.push_back(std::async(std::launch::async, &ThreadManagement::setLayers, this, std::vector<Layers>()));
        setFutures(std::move(futures));
    }
    void run() {
        RunEverything();
        for (auto& futurePromise : getFutures()) {
            futurePromise.wait();
        }
    }
private:
    using Layers = typename NetWorkInterface<T>::Layers;
};





int main()
{
    std::cout << "Hello World!\n";
}
