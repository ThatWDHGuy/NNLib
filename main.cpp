#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "nnlib.h"


TrainItem* processData(std::string file){
    TrainItem* item = new TrainItem();
    std::vector<int> in;
    std::vector<int> out;
    std::ifstream tFile(file, std::ios_base::in);
    float a;
    int c = 0;
    int numInputs = 2;
    while (tFile >> a)
    {
        if (c < numInputs){
            in.push_back((int)a);
        } else {
            out.push_back((int)a);
        }
        c++;
    }
    item->setInOut(in, out);
    return item;
}

int main(void){
    srand (time(NULL));
    NNLib nn = NNLib();
    std::vector<int> i{2,2,1}; 
    nn.setLayers(&i);
    nn.makeLinks(nn.ALL);
    nn.randWeightBias();
    nn.printNet();
    nn.loadTrainingSet(&processData);
    /*for (int i = 0; i < nn.getTraining()->size(); i++){
        std::cout<<i<<std::endl;
        nn.getTraining()->at(i)->print();
    }*/
    std::vector<float> in{0.0, 0.0}; 
    nn.evaluateInput(&in);
    std::vector<float> in2{1.0, 0.0}; 
    nn.evaluateInput(&in2);
    std::vector<float> in3{0.0, 1.0}; 
    nn.evaluateInput(&in3);
    std::vector<float> in4{1.0, 1.0}; 
    nn.evaluateInput(&in4);
    return 0;
}