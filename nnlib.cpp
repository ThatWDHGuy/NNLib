#include "nnlib.h"
#include <iostream>
#include <filesystem>
#include <string>
namespace fs = std::filesystem;

NNLib::NNLib(){

}

void NNLib::setLayers(std::vector<uint8_t>* lays){
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
    for (int l = 0; l < net.size(); l++){
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
            std::cout<<l<<" "<<n<<" : "<<neu<<"\n\t"<<neu->getWeight()<<"\n\tfw:"<<std::endl;
            for (int i = 0; i < neu->getFw()->size(); i++){
                std::cout<<"\t\t"<<neu->getFw()->at(i)<<" "<<neu->getBias()->at(i)<<std::endl;
            }
            std::cout<<"\tbw:"<<std::endl;
            for (int i = 0; i < neu->getBw()->size(); i++){
                std::cout<<"\t\t"<<neu->getBw()->at(i)<<std::endl;
            }
        }
    }
}

void NNLib::loadTrainingSet(){//TrainItem func(void)){
    std::string path = "training";
    for (const auto & entry : fs::directory_iterator(path))
        std::cout << entry.path() << std::endl;
}

void NNLib::trainNet(float maxError, int maxIterations){

}