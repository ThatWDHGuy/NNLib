#include <vector>

class TrainItem{
    private:
        std::vector<float> inputs;
        std::vector<float> outputs;

    public:
        TrainItem();
        void setInOut(std::vector<float> input, std::vector<float> output);
        std::vector<float>* getInputs();
        std::vector<float>* getOutputs();
        void print();
};