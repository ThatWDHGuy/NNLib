#include <iostream>
#include <stdint.h>
#include "nnlib.h"

int main(void){
    NNLib nn = NNLib();
    std::vector<uint8_t> i{16,32,64,128,8};
    std::cout<<"Starting net creation"<<std::endl;
    nn.setLayers(&i);
    std::cout<<"Layers made"<<std::endl;
    nn.makeLinks(nn.ALL);
    std::cout<<"Links made"<<std::endl;
    nn.randWeightBias();
    std::cout<<"Weights and bias set"<<std::endl;
    return 0;
}