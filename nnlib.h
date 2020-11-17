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
        TrainItem* getTrainItem(int i);
        float totalDatasetError();
        void stepGradCalc(std::vector<float>* inputs, std::vector<float>* outputs);
        void doABackProp();
        void stepByGradient();
        void resetAllDeltas();
        
    public:
        NNLib();
        std::vector<std::vector<Neuron*>> getNet();
        void setLayers(std::vector<int>* lays);
        void randWeightBias(float min, float max);
        enum Mode {ALL};
        void makeLinks(Mode m);
        void printNet();
        void loadTrainingSet(TrainItem* func(std::string));
        void loadTrainingFile(std::vector<TrainItem*>* func(std::string));
        void trainNet(float maxError, int maxIterations);
        std::vector<TrainItem*>* getTraining();
        float forwardProp(std::vector<float>* inputs, std::vector<float>* outputs);
        void getResults(std::vector<float>* inputs);
};