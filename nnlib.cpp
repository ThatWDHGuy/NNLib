#include "nnlib.h"
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

NNLib::NNLib(){

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

void NNLib::trainNet(float maxError, int maxIterations){
    float totalError;
    int iter = 0;
    while (totalError > maxError || iter < maxIterations){
        totalError = doABackProp(getRandTrain());
    }
}

float NNLib::doABackProp(TrainItem* trainData){
    
    return 0;
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

void NNLib::evaluateInput(std::vector<float>* inputs){
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
    for (int i = 0; i < net.at(net.size()-1).size(); i++){
        Neuron* neu = net.at(net.size()-1).at(i);
        neu->setVal(neu->getVal() + neu->getBias());
        std::cout<<neu->getVal()<<std::endl;
    }
}