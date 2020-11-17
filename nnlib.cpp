#include "nnlib.h"
#include <iostream>
#include <filesystem>
#include <cmath>
#include <chrono>

namespace fs = std::filesystem;

NNLib::NNLib(){
    
}

float NNLib::activation(float x)
{
    return 1.0/(1.0 + std::exp(-x));
}

// derivative of activation function
float NNLib::d_activation(float x){
    return (1.0 - activation(x))*activation(x);
}  

void NNLib::setLayers(std::vector<int>* lays){
    for (int l = 0; l < lays->size(); l++){
        std::vector<Neuron*> layer;
        for (int n = 0; n < lays->at(l); n++){
            Neuron* neu = new Neuron();
            layer.push_back(neu);
        }
        net.push_back(layer);
    }
}

std::vector<std::vector<Neuron*>> NNLib::getNet(){
    return net;
}

void NNLib::randWeightBias(float min, float max){
    for (int n = 0; n < net.at(0).size(); n++){
        Neuron *neu = net.at(0).at(n);
        neu->randInitBias(min, max);
    }
    for (int l = 1; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron *neu = net.at(l).at(n);
            neu->randInitWeightBias(min, max);
        }
    }
}

void NNLib::makeLinks(Mode m){
    switch (m){
    case ALL:
        linkAllForward();
        break;
    }
}

void NNLib::linkAllForward(){
    //link all nodes forward
    for (int l = 0; l < net.size()-1; l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron *current = net.at(l).at(n);
            for (int nn = 0; nn < net.at(l+1).size(); nn++){
                Neuron *next = net.at(l+1).at(nn);
                current->getFw()->push_back(next);
                next->getBw()->push_back(current);
            }
        }
    }
}

void NNLib::printNet(){
    for (int l = 0; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron *neu = net.at(l).at(n);
            std::cout<<l<<","<<n<<":\n Val: "<<neu->getVal()<<"\n Bias:\n  "<<neu->getBias()<<"\n Weights:"<<std::endl;
            for (int i = 0; i < neu->getFw()->size(); i++){
                std::cout<<"  "<<neu->getWeights()->at(i)<<std::endl;
            }
            std::cout<<std::endl;
        }
    }
}

void NNLib::loadTrainingSet(TrainItem* func(std::string)){
    std::string path = "training";
    for (const auto & entry : fs::directory_iterator(path)){
        //std::cout << entry.path() << std::endl;
        training.push_back(func(entry.path().string()));
    }
}

void NNLib::loadTrainingFile(std::vector<TrainItem*>* func(std::string)){
    std::string path = "training";
    
    for (const auto & entry : fs::directory_iterator(path)){
        
        std::vector<TrainItem*>* returned = func(entry.path().string());
        std::cout<<returned->size()<<std::endl;
        for (int i = 0; i < returned->size(); i++){
            training.push_back(returned->at(i));
        }
    }
}

TrainItem* NNLib::getRandTrain(){
    int a = ((float)rand()/RAND_MAX)*training.size();
    return training.at(a);
}

TrainItem* NNLib::getTrainItem(int i){
    return training.at(i);
}

float NNLib::totalDatasetError(){ // sum of errors for all rows of train data
	float tde = 0.0;
	for (int i = 0 ; i < training.size(); i++){
        TrainItem* item = training.at(i);
	    tde = tde + forwardProp(item->getInputs(), item->getOutputs());
	}
	return tde;
}

void NNLib::trainNet(float maxError, int maxIterations, float lr, bool backprop, bool printTimes){
    int iter = 0;
    dStep = 0.01;  // step to estimate gradient
    learningRate = lr;
    while (( iter < maxIterations) && (totalDatasetError() > maxError) ){ 
        auto start = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds;
        
        TrainItem* item = getRandTrain();
        if (printTimes){
            for (int i = 0 ; i < 8; i++){
                for ( int j = 0 ; j < 8;j++){
                    if (item->getInputs()->at(i*8 + j) > 0.0){
                        std::cout<<"0";
                    } else {
                        std::cout<<"-";
                    }
                }
                std::cout<<std::endl;
            }
        }
        auto printedInputs = std::chrono::steady_clock::now();
        elapsed_seconds = printedInputs-start;
        if (printTimes) std::cout << "Time to print: " << elapsed_seconds.count() << "s\n";

        forwardProp(item->getInputs(), item->getOutputs());

        auto fp = std::chrono::steady_clock::now();
        elapsed_seconds = fp-printedInputs;
        if (printTimes)std::cout << "Time to FP: " << elapsed_seconds.count() << "s\n";

        if (backprop) doABackProp();
        else stepGradCalc(item->getInputs(), item->getOutputs()); //SLOW
        

        auto gradCalc = std::chrono::steady_clock::now();
        elapsed_seconds = gradCalc-fp;
        if (printTimes)std::cout << "Time to gradCalc: " << elapsed_seconds.count() << "s\n";

        stepByGradient();

        auto stepGrad = std::chrono::steady_clock::now();
        elapsed_seconds = stepGrad-gradCalc;
        if (printTimes)std::cout << "Time to stepGrad: " << elapsed_seconds.count() << "s\n";

        //printNet();
        std::cout<<"step: "<<iter<<", Total dataset error: "<< totalDatasetError()<<std::endl;
        iter++;

        auto end = std::chrono::steady_clock::now();
        elapsed_seconds = end-start;
        if (printTimes) std::cout << "Total step time: " << elapsed_seconds.count() << "s\n";
    }
}

void NNLib::resetAllDeltas(){
     for (int l = 0; l < net.size(); l++){
        for (int i = 0; i < net.at(l).size(); i++){
            Neuron* neuI = net.at(l).at(i);
            for (int i = 0; i < net.at(l).size(); i++){
                neuI->setD_Bias(0);
            }
            for (int w = 0; w < neuI->getWeights()->size(); w++){
                neuI->setD_Weight(w, 0);
            }
        }
    }
}

void NNLib::stepGradCalc(std::vector<float>* inputs, std::vector<float>* outputs){

    for (int l = 0; l < net.size(); l++){
        for (int i = 0; i < net.at(l).size(); i++){
            float step = 0.001;
            Neuron* neuI = net.at(l).at(i);
            float oldError;
            float oldVal;
            float newError;

            oldError = forwardProp(inputs, outputs);
            if (i != 0){
                oldVal = neuI->getBias();
                neuI->setBias(oldVal + step);
                newError = forwardProp(inputs, outputs);
                neuI->setBias((newError-oldError)/step);
                neuI->setBias(oldVal);
            }
            for (int w = 0; w < neuI->getWeights()->size(); w++){
                oldVal = neuI->getWeight(w);
                neuI->setWeight(w, oldVal + step);
                newError = forwardProp(inputs, outputs);
                neuI->setD_Weight(w, (newError-oldError)/step);
                neuI->setWeight(w, oldVal);
            }
        }
    }

}

void NNLib::doABackProp(){
    int l = net.size()-1;
    //std::cout<<l<<std::endl;
    for (int i = 0; i < net.at(l).size(); i++){
        Neuron* neuI = net.at(l).at(i);
        neuI->setDelta(d_activation(neuI->getVal())*neuI->getError()*2);
        neuI->setD_Bias(neuI->getDelta());
            
        for (int j = 0; j < net.at(l-1).size(); j++){
            Neuron* neuJ = net.at(l-1).at(j);
            neuJ->setD_Weight(i, neuI->getDelta() * neuJ->getVal());
		}

	}
    //std::cout<<l<<std::endl;
    for (l = net.size()-2; l >= 1; l--){
        std::cout<<l<<std::endl;
        for (int j = 0; j < net.at(l).size(); j++){
            Neuron* neuJ = net.at(l).at(j);
            float sumDelta = 0;
            for (int i = 0; i < net.at(l+1).size(); i++){
                Neuron* neuI = net.at(l+1).at(i);
                sumDelta += neuI->getDelta()*neuJ->getWeight(i);
            }
            
            neuJ->setDelta(sumDelta*d_activation(neuJ->getNet()));
            neuJ->setD_Bias(neuJ->getDelta());
            for (int k = 0; k < net.at(l-1).size(); k++){
                Neuron* neuK = net.at(l-1).at(k);
                neuK->setD_Weight(j, neuJ->getDelta() * neuK->getVal());
            }
        }
    }
}

void NNLib::stepByGradient(){
	//bias
	for (int l = 1; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron* neu = net.at(l).at(n);
		    neu->setBias(neu->getBias() + neu->getD_Bias()*learningRate);
	    }
    }
	
	//weights
	for (int l = 0; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron* neu = net.at(l).at(n);
		    for (int w = 0; w < neu->getWeights()->size();w++)
                neu->setWeight(w, neu->getWeight(w) + neu->getD_Weight(w)*learningRate);
	    }
    }
}

std::vector<TrainItem*>* NNLib::getTraining(){
    return &training;
}

void NNLib::resetVals(){
    for (int l = 0; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            net.at(l).at(n)->setVal(0);
        }
    }
}

float NNLib::forwardProp(std::vector<float>* inputs, std::vector<float>* outputs){
    
    int l = 0;
    for (int i = 0; i < net.at(0).size(); i++){
        Neuron* neu = net.at(0).at(i);
        neu->setVal(inputs->at(i));
    }

    for (l = 1; l < net.size(); l++){
        for (int i = 0; i < net.at(l).size(); i++){
            Neuron* neuI = net.at(l).at(i);
            neuI->setNet(neuI->getBias());
            for (int j = 0; j < net.at(l-1).size(); j++){
                Neuron* neuJ = net.at(l-1).at(j);
                neuI->addNet(neuJ->getWeight(i)*neuJ->getVal());
            }
            neuI->setVal(activation(neuI->getNet()));
        }
    }

    l = net.size()-1;
    float cost = 0;
    for (int i = 0; i < net.at(l).size(); i++){
        Neuron* neu = net.at(l).at(i);
        if (outputs->size() != 0){
            float e = neu->getVal() - outputs->at(i);
            neu->setError(e*e);
            cost += neu->getError();
        }
    }
    return cost;
}

void NNLib::getResults(std::vector<float>* inputs){
    std::vector<float> out;
    forwardProp(inputs, &out);
    for (int i = 0; i < net.at(net.size()-1).size(); i++){
        Neuron* neu = net.at(net.size()-1).at(i);
        std::cout<<neu->getVal()<<std::endl;
    }
}

void NNLib::getResults(int num){
    std::vector<float> out;
    forwardProp(training.at(num)->getInputs(), &out);
    TrainItem* item = training.at(num);
        for (int i = 0 ; i < 8; i++){
            for ( int j = 0 ; j < 8;j++){
                if (item->getInputs()->at(i*8 + j) > 0.0){
                    std::cout<<"0";
                } else {
                    std::cout<<"-";
                }
            }
            std::cout<<std::endl;
        }
    for (int i = 0; i < net.at(net.size()-1).size(); i++){
        Neuron* neu = net.at(net.size()-1).at(i);
        float f = neu->getVal() > 0.9 ? 1 : (neu->getVal() < 0.1 ? 0 :neu->getVal());
        std::cout<<f<<std::endl;
    }
}

