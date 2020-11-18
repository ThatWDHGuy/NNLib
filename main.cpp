#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "nnlib.h"

std::vector<TrainItem*> items;

std::vector<TrainItem*>* processData(std::string file){
    items = {};
    TrainItem* item = new TrainItem();
    std::vector<float> in;
    std::vector<float> out;
    std::ifstream tFile(file, std::ios_base::in);
    float a;
    int c = 0;
    int numInputs = 64;
    int numOutputs = 8;
    while (tFile >> a){
        if (c == numInputs+numOutputs){
            c = 0;
            item->setInOut(in, out);
            items.push_back(item);
            item = new TrainItem();
            in.clear();
            out.clear();
        }
        if (c < numInputs){
            in.push_back((float)a);
        } else {
            out.push_back((float)a);
        }
        c++;
    }
    item->setInOut(in, out);
    items.push_back(item);
    item = new TrainItem();
    in.clear();
    out.clear();
    return (&items);
}

void displayData(std::vector<float>* input){
  for (int i = 0 ; i < 8; i++){
    for ( int j = 0 ; j < 8;j++){
       if (input->at(i*8 + j) > 0.0){
          std::cout<<"0";
       } else {
         std::cout<<"-";
       }
    }
    std::cout<<std::endl;
  }
}

int main(void){
    NNLib nn = NNLib();
    nn.setDataDisplay(&displayData);
    nn.loadTrainingFile(&processData);
    nn.cliMenu();
    return 0;
}