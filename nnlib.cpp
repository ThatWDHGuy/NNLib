#include "nnlib.h"
#include <iostream>

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
        setAllForward();
        break;
    }
}


void NNLib::setAllForward(){
    for (int l = 0; l < net.size()-1; l++){
        for (int nStart = 0; nStart < net.at(l).size(); nStart++){
            Neuron *start = net.at(l).at(nStart);
            for (int nEnd = 0; nEnd < net.at(l+1).size(); nEnd++){
                Neuron *end = net.at(l+1).at(nEnd);
                start->addForward(end);
                end->addBackward(start);
            }
        }
    }
}