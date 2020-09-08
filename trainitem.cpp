#include "trainitem.h"
#include <iostream>

TrainItem::TrainItem(){
}

void TrainItem::setInOut(std::vector<float> input, std::vector<float> output){
    inputs = input;
    outputs = output;
}

std::vector<float>* TrainItem::getInputs(){
    return &inputs;
}

std::vector<float>* TrainItem::getOutputs(){
    return &outputs;
}

void TrainItem::print(){
    std::cout<<"in:"<<std::endl;
    for (int i = 0; i < inputs.size(); i++){
        std::cout<<"\t"<<inputs.at(i)<<std::endl;
    }
    std::cout<<"out:"<<std::endl;
    for (int i = 0; i < outputs.size(); i++){
        std::cout<<"\t"<<outputs.at(i)<<std::endl;
    }
}