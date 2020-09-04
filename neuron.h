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
        void addForward(Neuron *fw);
        void addBackward(Neuron *bw);
        void randInitWeightBias();
        float getBias(uint16_t i);
        float getWeight();

};