#include "nnlib.h"
#include <iostream>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <regex>
#include <climits>

namespace fs = std::filesystem;

NNLib::NNLib(){
    srand (time(NULL));
    layerNums = {64,16,8};
    learningRate = -0.25;
    maxTrainError = 4;
    maxIter = 20000;
    backProp = false;
    printTrainData = false;
    linkMode = ALL;
    minWeiBias = -1.0f;
    maxWeiBias = 1.0f;
    roundOutput = false;
}

void NNLib::cliMenu(){
    while (1){
        int i = 1,
        initOpt = i++,
        trainOpt = i++,
        evaluateOpt = i++,
        configureOpt = i++,
        checkDataOpt = i++,
        saveOpt = i++,
        loadOpt = i++;

        std::string layerStr = "(";
        for (int i = 0; i < net.size(); i++){
            layerStr.append(std::to_string(net.at(i).size()));
            if (i != net.size()-1)
                layerStr.append(" ");
        }
        if (net.size() == 0)
            layerStr.append("Uninitalized");
        layerStr.append(") -> (");
        for (int i = 0; i < layerNums.size(); i++){
            layerStr.append(std::to_string(layerNums.at(i)));
            if (i != layerNums.size()-1)
                layerStr.append(" ");
        }
        layerStr.append(")");

        

        std::cout<<\
        initOpt<<") Init Net - "<<layerStr<<"\n"<<\
        trainOpt<<") Train\n"<<\
        evaluateOpt<<") Evaluate\n"<<\
        configureOpt<<") Configure Net\n"<<\
        checkDataOpt<<") Check Dataset\n"<<\
        saveOpt<<") Save Net\n"<<\
        loadOpt<<") Load Net\n"<<\
        "x) Exit\n"<<\
        "Input:";

        std::string input;
        std::cin.clear();
        std::cin >> input;
        char a = input.at(0);
        a -= '0';

        if (initOpt == a) { /* init */
            initNet();
        } else if (trainOpt == a) { /* train */
            menuTrainOption();
        } else if (evaluateOpt == a){ /*evaluate*/
            menuEvalOption();
        } else if (configureOpt == a){ /*configure net*/
            menuConfigureOption();
        } else if (checkDataOpt == a){ /*check dataset*/
            menuCheckDatasetOption();
        } else if (saveOpt == a){ /*save net*/

        } else if (loadOpt == a){ /*load net*/

        } else if ('x' == a + '0'){ /*exit*/
            break;
        }else { /* invalid */
            std::cout<<"Invalid Entry"<<std::endl;
        }
        std::cout<<std::endl;
    }
}

void NNLib::initNet(){
    setLayers();
    makeLinks();
    randWeightBias();
}

void NNLib::menuTrainOption(){
    
    std::cout<<"\n---Training---\n";
    if (net.size() < 2){
        std::cout<<"Net not Initalized, not net exists, please initalize\n";
        return;
    }
    std::string init;
    bool valid = false;
    std::cin.ignore();
    while (!valid){
        std::cout<<"Re-randomise weights and bias'? (Y/n)";std::getline(std::cin, init);
        if (init.length() == 0){
            valid = true;
            randWeightBias();
        } else {
            switch (init.at(0)){
            case 'Y':
            case 'y':
                valid = true;
                randWeightBias();
                break;
            case 'N':
            case 'n':
                valid = true;
                break;
            }
        }
    }
    trainNet();
}

void NNLib::menuEvalOption(){
    
    std::cout<<"\n---Evaluating---\n";
    if (net.size() < 2){
        std::cout<<"Net not Initalized, not net exists, please initalize\n";
    } else {
        while (1) {
            std::string input;
            std::cout<<"Input from dataset: ";
            std::cin.clear();
            std::cin >> input;
            std::regex reg("^[0-9]{1,10}$");
            char a = input.at(0);
            if (std::regex_match(input, reg)) { /* evaluate */
                int num = std::stoi(input);
                if (num >=0 && num < training.size()){
                    getResults(num);
                } else {
                    std::cout<<"input out of bounds (0-"<<training.size()-1<<")"<<std::endl;
                }
            } else if (a == 'x'){ /* exit */
                break;
            }else { /* invalid */
                std::cout<<"Invalid Entry"<<std::endl;
            }
        }
    }
}

void NNLib::menuConfigureOption(){
    while (1){
        int i = 1,
        netLayoutOpt = i++,
        backpropOpt = i++,
        maxTrainErrorOpt = i++,
        maxIterationsOpt = i++,
        LearningrateOpt = i++,
        minRandWeiBiasOpt = i++,
        maxRandWeiBiasOpt = i++,
        printTrainDataOpt = i++,
        roundOutputOpt = i++;


        std::string layerStr = "(";
        for (int i = 0; i < layerNums.size(); i++){
            layerStr.append(std::to_string(layerNums.at(i)));
            if (i != layerNums.size()-1)
                layerStr.append(" ");
        }
        layerStr.append(")");

        

        std::cout<<\
        netLayoutOpt<<") Edit Net Layout: "<<layerStr<<"\n"<<\
        backpropOpt<<") Toggle BackProp: "<<(backProp?"True":"False")<<"\n"<<\
        maxTrainErrorOpt<<") Edit Max Training Error: "<<maxTrainError<<"\n"<<\
        maxIterationsOpt<<") Edit Max Training Iterations: "<<maxIter<<"\n"<<\
        LearningrateOpt<<") Edit Learning Rate: "<<learningRate<<"\n"<<\
        minRandWeiBiasOpt<<") Edit Min Random Weight Bias: "<<minWeiBias<<"\n"<<\
        maxRandWeiBiasOpt<<") Edit Max Random Weight Bias: "<<maxWeiBias<<"\n"<<\
        printTrainDataOpt<<") Toggle Print Training Data: "<<(printTrainData?"True":"False")<<"\n"<<\
        roundOutputOpt<<") Toggle Output Rounding: "<<(roundOutput?"True":"False")<<"\n"<<\
        "x) Exit\n"<<\
        "Input:";

        std::string input;
        std::cin.clear();
        std::cin >> input;
        char a = input.at(0);
        a -= '0';

        if (netLayoutOpt == a) { /* Edit Net Layout */
            setNewLayerSizes();
        } else if (backpropOpt == a) { /* Toggle BackProp */
            backProp = !backProp;
        } else if (maxTrainErrorOpt == a) { /* Edit Max Training Error */
            float f = getFloatInput();
            if (f != INFINITY)
                maxTrainError = f;
        } else if (maxIterationsOpt == a) { /* Edit Max Training Iterations */
            int i = getIntInput();
            if (i != INT_MAX)
                maxIter = i;
        } else if (LearningrateOpt == a) { /* Edit Learning Rate */
            float f = getFloatInput();
            if (f != INFINITY)
                learningRate = f;
        } else if (minRandWeiBiasOpt == a) { /* Edit Min Random Weight Bias */
            float f = getFloatInput();
            if (f != INFINITY)
                maxTrainError = f;
        } else if (maxRandWeiBiasOpt == a) { /* Edit Max Random Weight Bias */
            float f = getFloatInput();
            if (f != INFINITY)
                maxTrainError = f;
        } else if (printTrainDataOpt == a) { /* Toggle Print Training Data */
            printTrainData = !printTrainData;
        } else if (roundOutputOpt == a) { /* Toggle Output rounding Data */
            roundOutput = !roundOutput;
        } else if ('x' == a + '0'){ /*exit*/
            break;
        }else { /* invalid */
            std::cout<<"Invalid Entry"<<std::endl;
        }
        std::cout<<std::endl;
    }
}

void NNLib::setNewLayerSizes(){
    int layerCount = 1;
    std::vector<int> lays = {};
    while (1) {
        std::string input;
        std::cout<<"Enter number of neurons in layer "<<layerCount++<<": ('x' to end layer input)";
        std::cin.clear();
        std::cin >> input;
        std::regex reg("^[1-9]{1}[0-9]*$");
        char a = input.at(0);
        if (std::regex_match(input, reg)) { /* evaluate */
            lays.push_back(std::stoi(input));
        } else if (a == 'x'){ /* exit */
            break;
        }else { /* invalid */
            std::cout<<"Invalid Entry"<<std::endl;
        }
    }
    if (lays.size() != 0)
        layerNums = lays;
}

int NNLib::getIntInput(){
    while (1) {
        std::string input;
        std::cout<<"Input an Int: ";
        std::cin.clear();
        std::cin >> input;
        std::regex reg("^-?[0-9]+$");
        char a = input.at(0);
        if (std::regex_match(input, reg)) { /* evaluate */
            return std::stoi(input);
        } else if (a == 'x'){ /* exit */
            break;
        }else { /* invalid */
            std::cout<<"Invalid Entry"<<std::endl;
        }
    }
    return INT_MAX;
}

float NNLib::getFloatInput(){
    while (1) {
        std::string input;
        std::cout<<"Input a Float: ";
        std::cin.clear();
        std::cin >> input;
        std::regex reg("^-?[0-9.]+$");
        char a = input.at(0);
        if (std::regex_match(input, reg)) { /* evaluate */
            return std::stof(input);
        } else if (a == 'x'){ /* exit */
            break;
        }else { /* invalid */
            std::cout<<"Invalid Entry"<<std::endl;
        }
    }
    return INFINITY;
}

void NNLib::menuCheckDatasetOption(){
    while (1) {
        std::string input;
        std::cout<<"\n---DataSet Validation---\nInput from dataset: ";
        std::cin.clear();
        std::cin >> input;
        std::regex reg("^[0-9]{1,10}$");
        char a = input.at(0);
        if (std::regex_match(input, reg)) { /* evaluate */
            int num = std::stoi(input);
            if (num >=0 && num < training.size()){
                dataDisplay(training.at(num)->getInputs());
            } else {
                std::cout<<"input out of bounds (0-"<<training.size()-1<<")"<<std::endl;
            }
        } else if (a == 'x'){ /* exit */
            break;
        }else { /* invalid */
            std::cout<<"Invalid Entry"<<std::endl;
        }
    }
}

void NNLib::setDataDisplay(void func(std::vector<float>* inputs)){
    dataDisplay = func;
}

float NNLib::activation(float x)
{
    return 1.0/(1.0 + std::exp(-x));
}

float NNLib::d_activation(float x){
    return (1.0 - activation(x))*activation(x);
}  

void NNLib::setLayers(){
    net.clear();
    for (int l = 0; l < layerNums.size(); l++){
        std::vector<Neuron*> layer;
        for (int n = 0; n < layerNums.at(l); n++){
            Neuron* neu = new Neuron();
            layer.push_back(neu);
        }
        net.push_back(layer);
    }
}

std::vector<std::vector<Neuron*>> NNLib::getNet(){
    return net;
}

void NNLib::randWeightBias(){
    for (int n = 0; n < net.at(0).size(); n++){
        Neuron *neu = net.at(0).at(n);
        neu->randInitBias(minWeiBias, maxWeiBias);
    }
    for (int l = 1; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron *neu = net.at(l).at(n);
            neu->randInitWeightBias(minWeiBias, maxWeiBias);
        }
    }
}

void NNLib::makeLinks(){
    switch (linkMode){
        case ALL:
            linkAllForward();
            break;
    }
}

void NNLib::linkAllForward(){
    //link all nodes forward
    for (int l = 0; l < net.size()-1; l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron *current = net.at(l).at(n);
            for (int nn = 0; nn < net.at(l+1).size(); nn++){
                Neuron *next = net.at(l+1).at(nn);
                current->getFw()->push_back(next);
                next->getBw()->push_back(current);
            }
        }
    }
}

void NNLib::printNet(){
    for (int l = 0; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron *neu = net.at(l).at(n);
            std::cout<<l<<","<<n<<":\n Val: "<<neu->getVal()<<"\n Bias:\n  "<<neu->getBias()<<"\n Weights:"<<std::endl;
            for (int i = 0; i < neu->getFw()->size(); i++){
                std::cout<<"  "<<neu->getWeights()->at(i)<<std::endl;
            }
            std::cout<<std::endl;
        }
    }
}

void NNLib::loadTrainingSet(TrainItem* func(std::string)){
    std::string path = "training";
    for (const auto & entry : fs::directory_iterator(path)){
        //std::cout << entry.path() << std::endl;
        training.push_back(func(entry.path().string()));
    }
}

void NNLib::loadTrainingFile(std::vector<TrainItem*>* func(std::string)){
    std::string path = "training";
    
    for (const auto & entry : fs::directory_iterator(path)){
        
        std::vector<TrainItem*>* returned = func(entry.path().string());
        std::cout<<returned->size()<<std::endl;
        for (int i = 0; i < returned->size(); i++){
            training.push_back(returned->at(i));
        }
    }
}

TrainItem* NNLib::getRandTrain(){
    int a = ((float)rand()/RAND_MAX)*training.size();
    return training.at(a);
}

TrainItem* NNLib::getTrainItem(int i){
    return training.at(i);
}

float NNLib::totalDatasetError(){ // sum of errors for all rows of train data
	float tde = 0.0;
	for (int i = 0 ; i < training.size(); i++){
        TrainItem* item = training.at(i);
	    tde = tde + forwardProp(item->getInputs(), item->getOutputs());
	}
	return tde;
}

void NNLib::trainNet(){
    int iter = 0;
    dStep = 0.01;  // step to estimate gradient
    while (( iter < maxIter) && (totalDatasetError() > maxTrainError) ){
        
        std::chrono::steady_clock::time_point start, printedInputs, fp, gradCalc, stepGrad, end;
        if (printTrainData) start = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds;
        
        TrainItem* item = getRandTrain();
        if (printTrainData){
            dataDisplay(item->getInputs());
            printedInputs = std::chrono::steady_clock::now();
            elapsed_seconds = printedInputs-start;
            std::cout << "Time to print: " << elapsed_seconds.count() << "s\n";
        }

        forwardProp(item->getInputs(), item->getOutputs());

        if (printTrainData){
            fp = std::chrono::steady_clock::now();
            elapsed_seconds = fp-printedInputs;
            std::cout << "Time to FP: " << elapsed_seconds.count() << "s\n";
        }

        if (backProp) doABackProp();
        else stepGradCalc(item->getInputs(), item->getOutputs()); //SLOW
        
        if (printTrainData){
            gradCalc = std::chrono::steady_clock::now();
            elapsed_seconds = gradCalc-fp;
            std::cout << "Time to gradCalc: " << elapsed_seconds.count() << "s\n";
        }

        stepByGradient();
        
        if (printTrainData){
            stepGrad = std::chrono::steady_clock::now();
            elapsed_seconds = stepGrad-gradCalc;
            std::cout << "Time to stepGrad: " << elapsed_seconds.count() << "s\n";
        }

        //printNet();
        std::cout<<"step: "<<iter<<", Total dataset error: "<< totalDatasetError()<<std::endl;
        iter++;
        if (printTrainData) {
            end = std::chrono::steady_clock::now();
            elapsed_seconds = end-start;
            std::cout << "Total step time: " << elapsed_seconds.count() << "s\n";
        }
    }
}

void NNLib::resetAllDeltas(){
     for (int l = 0; l < net.size(); l++){
        for (int i = 0; i < net.at(l).size(); i++){
            Neuron* neuI = net.at(l).at(i);
            for (int i = 0; i < net.at(l).size(); i++){
                neuI->setD_Bias(0);
            }
            for (int w = 0; w < neuI->getWeights()->size(); w++){
                neuI->setD_Weight(w, 0);
            }
        }
    }
}

void NNLib::stepGradCalc(std::vector<float>* inputs, std::vector<float>* outputs){

    for (int l = 0; l < net.size(); l++){
        for (int i = 0; i < net.at(l).size(); i++){
            float step = 0.001;
            Neuron* neuI = net.at(l).at(i);
            float oldError;
            float oldVal;
            float newError;

            oldError = forwardProp(inputs, outputs);
            if (i != 0){
                oldVal = neuI->getBias();
                neuI->setBias(oldVal + step);
                newError = forwardProp(inputs, outputs);
                neuI->setBias((newError-oldError)/step);
                neuI->setBias(oldVal);
            }
            for (int w = 0; w < neuI->getWeights()->size(); w++){
                oldVal = neuI->getWeight(w);
                neuI->setWeight(w, oldVal + step);
                newError = forwardProp(inputs, outputs);
                neuI->setD_Weight(w, (newError-oldError)/step);
                neuI->setWeight(w, oldVal);
            }
        }
    }

}

void NNLib::doABackProp(){
    int l = net.size()-1;
    for (int i = 0; i < net.at(l).size(); i++){
        Neuron* neuI = net.at(l).at(i);
        neuI->setDelta(d_activation(neuI->getVal())*neuI->getError()*2);
        neuI->setD_Bias(neuI->getDelta());
        std::cout<<neuI->getDelta()<<std::endl;
        for (int j = 0; j < net.at(l-1).size(); j++){
            Neuron* neuJ = net.at(l-1).at(j);
            neuJ->setD_Weight(i, neuI->getDelta() * neuJ->getVal());
		}

	}
    for (l = net.size()-2; l >= 1; l--){
        for (int j = 0; j < net.at(l).size(); j++){
            Neuron* neuJ = net.at(l).at(j);
            float sumDelta = 0;
            for (int i = 0; i < net.at(l+1).size(); i++){
                Neuron* neuI = net.at(l+1).at(i);
                sumDelta += neuI->getDelta()*neuJ->getWeight(i);
            }
            
            neuJ->setDelta(sumDelta*d_activation(neuJ->getNet()));
            neuJ->setD_Bias(neuJ->getDelta());
            for (int k = 0; k < net.at(l-1).size(); k++){
                Neuron* neuK = net.at(l-1).at(k);
                neuK->setD_Weight(j, neuJ->getDelta() * neuK->getVal());
            }
        }
    }
}

void NNLib::stepByGradient(){
	//bias
	for (int l = 1; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron* neu = net.at(l).at(n);
		    neu->setBias(neu->getBias() + neu->getD_Bias()*learningRate);
	    }
    }
	
	//weights
	for (int l = 0; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            Neuron* neu = net.at(l).at(n);
		    for (int w = 0; w < neu->getWeights()->size();w++)
                neu->setWeight(w, neu->getWeight(w) + neu->getD_Weight(w)*learningRate);
	    }
    }
}

std::vector<TrainItem*>* NNLib::getTraining(){
    return &training;
}

void NNLib::resetVals(){
    for (int l = 0; l < net.size(); l++){
        for (int n = 0; n < net.at(l).size(); n++){
            net.at(l).at(n)->setVal(0);
        }
    }
}

float NNLib::forwardProp(std::vector<float>* inputs, std::vector<float>* outputs){
    
    int l = 0;
    for (int i = 0; i < net.at(0).size(); i++){
        Neuron* neu = net.at(0).at(i);
        neu->setVal(inputs->at(i));
    }

    for (l = 1; l < net.size(); l++){
        for (int i = 0; i < net.at(l).size(); i++){
            Neuron* neuI = net.at(l).at(i);
            neuI->setNet(neuI->getBias());
            for (int j = 0; j < net.at(l-1).size(); j++){
                Neuron* neuJ = net.at(l-1).at(j);
                neuI->addNet(neuJ->getWeight(i)*neuJ->getVal());
            }
            neuI->setVal(activation(neuI->getNet()));
        }
    }

    l = net.size()-1;
    float cost = 0;
    for (int i = 0; i < net.at(l).size(); i++){
        Neuron* neu = net.at(l).at(i);
        if (outputs->size() != 0){
            float e = neu->getVal() - outputs->at(i);
            neu->setError(e*e);
            cost += neu->getError();
        }
    }
    return cost;
}

void NNLib::getResults(std::vector<float>* inputs){
    std::vector<float> out;
    forwardProp(inputs, &out);
    dataDisplay(inputs);
    for (int i = 0; i < net.at(net.size()-1).size(); i++){
        Neuron* neu = net.at(net.size()-1).at(i);
        std::cout<<neu->getVal()<<std::endl;
    }
}

void NNLib::getResults(int num){
    std::vector<float> out;
    forwardProp(training.at(num)->getInputs(), &out);
    dataDisplay(training.at(num)->getInputs());
    for (int i = 0; i < net.at(net.size()-1).size(); i++){
        Neuron* neu = net.at(net.size()-1).at(i);
        float f = roundOutput ? (neu->getVal() > 0.5 ? 1 : 0) : neu->getVal();
        std::cout<<f<<std::endl;
    }
}

void NNLib::saveNet(){

}

void NNLib::loadNet(){

}