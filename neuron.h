#include <vector>

class Neuron{
    private:
        std::vector<Neuron*> fw;
        std::vector<Neuron*> bw;
        std::vector<float> weights;
        float bias;
        float val;
        float net;
        float getRand(float min, float max);
        std::vector<float> d_weights;
        float d_bias;
        float delta;
        float error;

    public:
        Neuron();
        void addForward(Neuron *f);
        void addBackward(Neuron *b);
        void randInitWeightBias(float min, float max);
        void randInitBias(float min, float max);
        float getWeight(int i);
        float getBias();
        void setWeight(int i, float v);
        void setBias(float v);
        std::vector<Neuron*>* getFw();
        std::vector<Neuron*>* getBw();
        std::vector<float>* getWeights();
        void setVal(float v);
        void addVal(float v);
        float getVal();
        void setNet(float v);
        void addNet(float v);
        float getNet();
        float getD_Weight(int i);
        float getD_Bias();
        std::vector<float>* getD_Weights();
        void setD_Weight(int i, float v);
        void setD_Bias(float v);
        float getDelta();
        void setDelta(float v);
        float getError();
        void setError(float v);
};