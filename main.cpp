#include <iostream>
#include <stdint.h>
#include "nnlib.h"

int main(void){
    NNLib nn = NNLib();
    std::vector<uint8_t> i{16,32,64,128,8}; 
    nn.setLayers(&i);
    return 0;
}