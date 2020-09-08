#include <stdint.h>
#include <vector>
#include <string>
#include "neuron.h"
#include "trainitem.h"

class NNLib{
    private:
        float learningRate;
        float dStep;
        std::vector<std::vector<Neuron*>> net;
        std::vector<TrainItem*> training;

        float activation(float x);
        float d_activation(float x);
        void linkAllForward();
        void resetVals();
        TrainItem* getRandTrain();
        float totalDatasetError();
        void doABackProp();
        void stepByGradient();
        
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
        float forwardProp(std::vector<float>* inputs, std::vector<float>* outputs);
        void getResults(std::vector<float>* inputs);
};