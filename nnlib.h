#include <vector>
#include "neuron.h"
#include <stdint.h>

class NNLib{
    private:
        std::vector<std::vector<Neuron*>> net;
    public:
        NNLib();
        void setLayers(std::vector<uint8_t>* lays);
        void randWeightBias();
};