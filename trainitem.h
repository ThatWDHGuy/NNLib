class TrainItem{
    private:
        std::vector<int> inputs;
        std::vector<int> outputs;

    public:
        TrainItem(std::vector<int> input, std::vector<int> output);
        std::vector<int>* getInputs();
        std::vector<int>* getOutputs();
};