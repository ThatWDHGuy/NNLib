#include "nnlib.h"
#include <iostream>
#include <filesystem>
#include <cmath>

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

void NNLib::randWeightBias(){
    for (int n = 0; n < net.at(0).size(); n++){
        Neuron *neu = net.at(0).at(n);
        neu->randInitBias();
    }
    for (int l = 1; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron *neu = net.at(l).at(n);
            neu->randInitWeightBias();
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
            std::cout<<l<<" "<<n<<" : "<<neu<<"\n\t"<<neu->getBias()<<"\n\tfw:"<<std::endl;
            for (int i = 0; i < neu->getFw()->size(); i++){
                std::cout<<"\t\t"<<neu->getFw()->at(i)<<" "<<neu->getWeights()->at(i)<<std::endl;
            }
            std::cout<<"\tbw:"<<std::endl;
            for (int i = 0; i < neu->getBw()->size(); i++){
                std::cout<<"\t\t"<<neu->getBw()->at(i)<<std::endl;
            }
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

TrainItem* NNLib::getRandTrain(){
    int a = ((float)rand()/RAND_MAX)*training.size();
    return training.at(a);
}

float NNLib::totalDatasetError(){ // sum of errors for all rows of train data
	float tde = 0.0;
	for (int i = 0 ; i < training.size(); i++){
        TrainItem* item = training.at(i);
	    tde = tde + forwardProp(item->getInputs(), item->getOutputs());
	}
	return tde;
}

void NNLib::trainNet(float maxError, int maxIterations){
    float totalError;
    int iter = 0;
    dStep = 0.01;  // step to estimate gradient
    learningRate = 0.05;
    int iImage = 0;
    srand (time(NULL));  // seed random number generator
    while (( iter < maxIterations) && (totalDatasetError() > maxError) ){ 
        TrainItem* item = getRandTrain();
        forwardProp(item->getInputs(), item->getOutputs());
        doABackProp();
        stepByGradient();
        std::cout<<"step: "<<iter<<std::endl;
        std::cout<<" Total dataset error: "<< totalDatasetError()<<std::endl;
        iter++;
    }
}

void NNLib::doABackProp(){
    
    //for(int k = 0; k < nOutputs; k++){
    for (int i = 0; i < net.at(net.size()-1).size(); i++){
		//deltaOutput[k] = d_activation(netOutput[k])*currentError[k]*2;
        Neuron* neuI = net.at(net.size()-1).at(i);
        neuI->setDelta(d_activation(neuI->getVal())*neuI->getError()*2);
		//d_biasOutput[k] = deltaOutput[k];
        neuI->setD_Bias(neuI->getDelta());
		//for (int i = 0; i < nHiddenNeurons; i++){
            
        for (int j = 0; j < net.at(net.size()-2).size(); j++){
			//d_weightsOutput[k*nHiddenNeurons + i] = deltaOutput[k]*outHidden[i];
            Neuron* neuJ = net.at(net.size()-2).at(j);
            std::cout<<neuJ->getD_Weights()->size()<<" "<<i<<std::endl;
            neuJ->setD_Weight(i, neuI->getDelta() * neuJ->getVal());
		}

	}
	
	//for (int j = 0; j < nHiddenNeurons; j++){
    for (int l = net.size()-2; l >= 1; --l){
        for (int j = 0; j < net.at(l).size(); j++){
            Neuron* neuJ = net.at(l).at(j);
            //double sumDelta = 0;
            float sumDelta = 0;
            //for (int i = 0; i < nOutputs; i++){
            for (int i = 0; i < net.at(l+1).size(); i++){
                Neuron* neuI = net.at(l+1).at(i);
                //sumDelta += deltaOutput[i]*weightsOutput[j*nOutputs + i];
                sumDelta += neuI->getDelta()*neuI->getWeight(i);
            }
            //deltaHidden[j] = sumDelta * d_activation(netHidden[j]);
            neuJ->setDelta(sumDelta*d_activation(neuJ->getVal()));
            //d_biasHidden[j] = deltaHidden[j];
            neuJ->setD_Bias(neuJ->getDelta());
            //for (int i = 0; i < nInputs; i++){
            for (int k = 0; k < net.at(l-1).size(); k++){
                //d_weightsHidden[j*nInputs + i] = deltaHidden[j] * currentInputs[i];
                Neuron* neuK = net.at(l-1).at(k);
                neuK->setD_Weight(j, neuJ->getDelta() * neuK->getVal());
            }

        }
    }
}

void NNLib::stepByGradient(){
	//bias
	for (int l = 0; l < net.size(); l++){
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

float NNLib::forwardProp(std::vector<float>* inputs, std::vector<float>* outputs = new std::vector<float>){
    resetVals();
    for (int i = 0; i < net.at(0).size(); i++){
        Neuron* neu = net.at(0).at(i);
        if (i < inputs->size()){
            neu->setVal(inputs->at(i));
        } else {
            neu->setVal(0);
        }
    }
    for (int l = 0; l < net.size()-1; l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron* neu = net.at(l).at(n);
            neu->addVal((neu->getVal()) + (neu->getBias()));
            for (int nn = 0; nn < net.at(l+1).size(); nn++){
                net.at(l+1).at(nn)->addVal((neu->getVal()) * (neu->getWeight(nn)));
            }
        }
    }
    float cost = 0;
    for (int i = 0; i < net.at(net.size()-1).size(); i++){
        Neuron* neu = net.at(net.size()-1).at(i);
        neu->setVal(neu->getVal() + neu->getBias());
        neu->setError(neu->getVal() - outputs->at(i));
        if (outputs->size() != 0)
            cost += std::pow(neu->getError(), 2.0);
    }
    if (outputs->size() == 0)
        delete outputs;
    return cost;
}

void NNLib::getResults(std::vector<float>* inputs){
    forwardProp(inputs);
    for (int i = 0; i < net.at(net.size()-1).size(); i++){
        Neuron* neu = net.at(net.size()-1).at(i);
        std::cout<<neu->getVal()<<std::endl;
    }
}

