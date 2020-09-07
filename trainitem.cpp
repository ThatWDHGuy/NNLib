#include "trainitem.h"
#include <iostream>

TrainItem::TrainItem(){
}

void TrainItem::setInOut(std::vector<int> input, std::vector<int> output){
    inputs = input;
    outputs = output;
}

std::vector<int>* TrainItem::getInputs(){
    return &inputs;
}

std::vector<int>* TrainItem::getOutputs(){
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