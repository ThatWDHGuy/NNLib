#include <cstdlib>
#include "neuron.h"

#define MIN -3
#define MAX 3

Neuron::Neuron(){

}

void Neuron::addForward(Neuron *f){
    fw.push_back(f);
}

void Neuron::addBackward(Neuron *b){
    bw.push_back(b);
}

float Neuron::getBias(uint16_t i){
    return bias.at(i);
}

float Neuron::getWeight(){
    return weight;
}

void Neuron::randInitWeightBias(){
    float rand;
    for (int i = 0; i < fw.size(); i++){
        rand = MIN + static_cast <float> (std::rand()) /( static_cast <float> (RAND_MAX/(MAX-MIN)));
        bias.push_back(rand);
    }
    rand = MIN + static_cast <float> (std::rand()) /( static_cast <float> (RAND_MAX/(MAX-MIN)));
    weight = rand;
}