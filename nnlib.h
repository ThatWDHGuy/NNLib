#include <stdint.h>
#include <vector>
#include <string>
#include "neuron.h"
#include "trainitem.h"

class NNLib{
    private:
        std::vector<std::vector<Neuron*>> net;
        std::vector<TrainItem*> training;
        void linkAllForward();
        void resetVals();
        TrainItem* getRandTrain();
        float doABackProp(TrainItem* trainData);
    public:
        NNLib();
        void setLayers(std::vector<int>* lays);
        void randWeightBias();
        enum Mode {ALL};
        void makeLinks(Mode m);
        void printNet();
        void loadTrainingSet(TrainItem* func(std::string));
        void trainNet(float maxError, int maxIterations);
        std::vector<TrainItem*>* getTraining();
        void evaluateInput(std::vector<float>* inputs);
};