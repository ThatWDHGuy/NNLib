#include <vector>

class TrainItem{
    private:
        std::vector<int> inputs;
        std::vector<int> outputs;

    public:
        TrainItem();
        void setInOut(std::vector<int> input, std::vector<int> output);
        std::vector<int>* getInputs();
        std::vector<int>* getOutputs();
        void print();
};