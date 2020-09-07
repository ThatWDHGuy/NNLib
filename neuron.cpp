#include <iostream>
#include "neuron.h"
#include <cstdlib>
#include <ctime>

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

float Neuron::getWeight(int i){
    return weights.at(i);
}

float Neuron::getBias(){
    return bias;
}

std::vector<Neuron*>* Neuron::getFw(){
    return &fw;
}

std::vector<Neuron*>* Neuron::getBw(){
    return &bw;
}

std::vector<float>* Neuron::getWeights(){
    return &weights;
}

float Neuron::getRand(){
    float a = ((float)rand()/RAND_MAX)*6 - 3;
    return a;
}

void Neuron::randInitWeightBias(){
    float rand;
    for (int i = 0; i < fw.size(); i++){
        weights.push_back(getRand());
    }
    bias = getRand();
}

void Neuron::randInitBias(){
    float rand;
    for (int i = 0; i < fw.size(); i++){
        weights.push_back(getRand());
    }
    bias = 0;
}

void Neuron::setVal(float v){
    val = v;
}

void Neuron::addVal(float v){
    val += v;
}

float Neuron::getVal(){
    return val;
}