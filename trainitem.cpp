#include "trainitem.h"

TrainItem::TrainItem(std::vector<int> input, std::vector<int> output){
    inputs = input;
    outputs = output;
}

std::vector<int>* TrainItem::getInputs(){
    return &inputs;
}

std::vector<int>* TrainItem::getOutputs(){
    return $outputs;
}