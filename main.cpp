#include <iostream>
#include <stdint.h>
#include "nnlib.h"

int main(void){
    NNLib nn = NNLib();
    /*std::vector<uint8_t> i{2,2,1}; 
    nn.setLayers(&i);
    nn.makeLinks(nn.ALL);
    nn.randWeightBias();*/
    nn.loadTrainingSet();
    return 0;
}