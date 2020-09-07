#include <vector>
#include "neuron.h"
#include <stdint.h>
#include "trainitem.h"

class NNLib{
    private:
        std::vector<std::vector<Neuron*>> net;
        void linkAllForward();
    public:
        NNLib();
        void setLayers(std::vector<uint8_t>* lays);
        void randWeightBias();
        enum Mode {ALL};
        void makeLinks(Mode m);
        void printNet();
        void loadTrainingSet();//TrainItem func(void));
        void trainNet(float maxError, int maxIterations);
};