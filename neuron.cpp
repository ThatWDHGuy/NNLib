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

void Neuron::setWeight(int i, float v){
    weights.at(i) = v;
}

void Neuron::setBias(float v){
    bias = v;
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
    float a = ((float)rand()/RAND_MAX)*2 - 1;
    return a;
}

void Neuron::randInitWeightBias(){
    for (int i = 0; i < fw.size(); i++){
        weights.push_back(getRand());
        d_weights.push_back(0);
    }
    bias = getRand();
    d_bias = 0;
}

void Neuron::randInitBias(){
    for (int i = 0; i < fw.size(); i++){
        weights.push_back(getRand());
        d_weights.push_back(0);
    }
    bias = 0;
    d_bias = 0;
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

void Neuron::setNet(float v){
    net = v;
}

void Neuron::addNet(float v){
    net += v;
}

float Neuron::getNet(){
    return net;
}

std::vector<float>* Neuron::getD_Weights(){
    return &d_weights;
}

float Neuron::getD_Weight(int i){
    return d_weights.at(i);
}

float Neuron::getD_Bias(){
    return d_bias;
}

void Neuron::setD_Weight(int i, float v){
    d_weights.at(i) = v;
}

void Neuron::setD_Bias(float v){
    d_bias = v;
}

float Neuron::getDelta(){
    return delta;
}

void Neuron::setDelta(float v){
    delta = v;
}

float Neuron::getError(){
    return error;
}

void Neuron::setError(float v){
    error = v;
}