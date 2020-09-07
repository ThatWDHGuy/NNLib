#include <vector>

class Neuron{
    private:
        std::vector<Neuron*> fw;
        std::vector<Neuron*> bw;
        std::vector<float> weights;
        float bias;
        float val;
        float getRand();

    public:
        Neuron();
        void addForward(Neuron *f);
        void addBackward(Neuron *b);
        void randInitWeightBias();
        void randInitBias();
        float getWeight(int i);
        float getBias();
        std::vector<Neuron*>* getFw();
        std::vector<Neuron*>* getBw();
        std::vector<float>* getWeights();
        void setVal(float v);
        void addVal(float v);
        float getVal();
};