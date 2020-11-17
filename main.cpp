#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <regex>
#include "nnlib.h"

std::vector<TrainItem*> items;

std::vector<TrainItem*>* processData(std::string file){
    /* INDIVIUDAL FILE PER TRAIN
    TrainItem* item = new TrainItem();
    std::vector<float> in;
    std::vector<float> out;
    std::ifstream tFile(file, std::ios_base::in);
    float a;
    int c = 0;
    int numInputs = 2;
    while (tFile >> a)
    {
        if (c < numInputs){
            in.push_back((float)a);
        } else {
            out.push_back((float)a);
        }
        c++;
    }
    item->setInOut(in, out);
    return item;*/
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

void displayDigit(std::vector<float>* input){
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
    srand (time(NULL));  // seed random number generator
    NNLib nn = NNLib();
    std::vector<int> i{64,32,8}; 
    nn.setLayers(&i);
    nn.makeLinks(nn.ALL);
    nn.randWeightBias(-1.0, 1.0);

    /*Neuron* neu = nn.getNet().at(0).at(0);
    neu->setBias(0);
    neu->setWeight(0, 20);
    neu->setWeight(1, -20);
    neu = nn.getNet().at(0).at(1);
    neu->setBias(0);
    neu->setWeight(0, 20);
    neu->setWeight(1, -20);
    neu = nn.getNet().at(1).at(0);
    neu->setBias(-10);
    neu->setWeight(0, 20);
    neu = nn.getNet().at(1).at(1);
    neu->setBias(30);
    neu->setWeight(0, 20);
    neu = nn.getNet().at(2).at(0);
    neu->setBias(-30);*/

    //nn.printNet();
    nn.loadTrainingFile(&processData);
    
    nn.trainNet(1, 20000, -0.2, false, false);

    while (true) {
        std::string a;
        std::cout<<"Input: ";
        std::cin.clear();
        std::cin >> a;
        std::regex reg("^[0-9]{1,10}$");
        if (std::regex_match(a, reg)){
            nn.getResults(std::stoi(a));
        } else {
            std::cout<<"Invalid Entry"<<std::endl;
        }
        std::cout<<std::endl;
    }
    
    return 0;
}