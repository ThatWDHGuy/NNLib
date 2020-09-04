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