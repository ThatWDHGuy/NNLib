#include <vector>
#include <stdint.h>

class Neuron{
    private:
        std::vector<Neuron*> fw;
        std::vector<Neuron*> bw;
        std::vector<float> bias;
        float weight;

    public:
        Neuron();
        void addForward(Neuron *f);
        void addBackward(Neuron *b);
        void randInitWeightBias();
        float getBias(uint16_t i);
        float getWeight();
        std::vector<Neuron*>* getFw();
        std::vector<Neuron*>* getBw();
        std::vector<float>* getBias();
};