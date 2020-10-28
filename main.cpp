#include <iostream>
#include <fstream>
#include <cstdlib>
#include "nnlib.h"


TrainItem* processData(std::string file){
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
    return item;
}

int main(void){
    NNLib nn = NNLib();
    std::vector<int> i{2,2,1}; 
    nn.setLayers(&i);
    nn.makeLinks(nn.ALL);
    nn.randWeightBias();
    
    Neuron* neu = nn.getNet().at(0).at(0);
    neu->setBias(0);
    neu->setWeight(0, 1);
    neu->setWeight(1, -1);
    neu = nn.getNet().at(0).at(1);
    neu->setBias(0);
    neu->setWeight(0, 1);
    neu->setWeight(1, -1);
    neu = nn.getNet().at(1).at(0);
    neu->setBias(0.5);
    neu->setWeight(0, 1);
    neu = nn.getNet().at(1).at(1);
    neu->setBias(-1.5);
    neu->setWeight(0, 1);
    neu = nn.getNet().at(2).at(0);
    neu->setBias(1.5);

    nn.printNet();

    /*nn.randWeightBias();
    nn.printNet();
    nn.loadTrainingSet(&processData);
    
    nn.trainNet(0.01, 1000);*/
    /*for (int i = 0; i < nn.getTraining()->size(); i++){
        std::cout<<i<<std::endl;
        nn.getTraining()->at(i)->print();
    }*/

    nn.printNet();
    std::vector<float> in{0.0, 0.0}; 
    nn.getResults(&in);
    std::vector<float> in2{1.0, 0.0}; 
    nn.getResults(&in2);
    std::vector<float> in3{0.0, 1.0}; 
    nn.getResults(&in3);
    std::vector<float> in4{1.0, 1.0}; 
    nn.getResults(&in4);
    return 0;
}