#include "neuron.h"

Neuron::Neuron(){

}

void Neuron::addForward(Neuron *fw){

}

void Neuron::addBackward(Neuron *bw){

}

double Neuron::getBias(int i){
    return bias.at(i);
}

double Neuron::getWeight(){
    return weight;
}