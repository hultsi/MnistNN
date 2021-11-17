#include <iostream>
#include <array>
#include <string>

#include "mnistNN.h"
#include "statpack.h"
#include "mnistParser.h"

// TODO: Change all std::array<std::array<...>> to one dimensionals...
namespace mnistNN {
    float initLearnRate(std::string filePath) {
        std::ifstream input(filePath);
        float val;
        if (input.is_open()) {
            input >> val;   
        }
        input.close();
        return val;
    }

    void train_v1(std::string images, std::string labels, std::string initValRoot) {
        std::cout << "Initiating train_v1\n";
        float learnRate = 500;

        constexpr const int inputLayerSize = 784;
        constexpr const int hLayerN1 = 20;
        constexpr const int hLayerN2 = 10;
        constexpr const int outputN = 10;

        std::array<std::array<float, inputLayerSize>, hLayerN1> weights1; //28^2
        std::array<std::array<float, hLayerN1>, hLayerN2> weights2;
        std::array<std::array<float, hLayerN2>, outputN> weights3;
        std::array<float, hLayerN1> bias1;
        std::array<float, hLayerN2> bias2;
        std::array<float, outputN> bias3;

        // Randomize initial weights
        if (initValRoot == "") {
            std::cout << "Randomizing weights and biases\n";

            for (int k = 0; k < hLayerN1; k++) {
                for (int i = 0; i < inputLayerSize; i++) {
                    weights1[k][i] = statpack::randomFloat(0,1) - .5f;
                }
                for (int i = 0; i < hLayerN2; i++) {
                    weights2[i][k] = statpack::randomFloat(0,1) - .5f;
                }
            }
            for (int i = 0; i < hLayerN2; i++) {
                for (int k = 0; k < outputN; k++) {
                    weights3[k][i] = statpack::randomFloat(0,1) - .5f;
                }
            }
            // Randomize biases
            for (int i = 0; i < hLayerN1; i++) {
                bias1[i] = statpack::randomFloat(0,1) - .5f;
            }
            for (int i = 0; i < hLayerN2; i++) {
                bias2[i] = statpack::randomFloat(0,1) - .5f;
            }
            for (int i = 0; i < outputN; i++) {
                bias3[i] = statpack::randomFloat(0,1) - .5f;
            }
        } else {
            initWeight<inputLayerSize, hLayerN1>(weights1, initValRoot + "/weights1.txt");
            initWeight<hLayerN1, hLayerN2>(weights2, initValRoot + "/weights2.txt");
            initWeight<hLayerN2, outputN>(weights3, initValRoot + "/weights3.txt");
            initBias<hLayerN1>(bias1, initValRoot + "/biases1.txt");
            initBias<hLayerN2>(bias2, initValRoot + "/biases2.txt");
            initBias<outputN>(bias3, initValRoot + "/biases3.txt");
            
            float lrate = mnistParser::initLearnRate(initValRoot + "/learnrate.txt");
            if (lrate != 0) {
                learnRate = lrate;
            }
        }
        std::array<float, outputN> result;
        std::array<float, outputN> targetResult;
        std::array<float, hLayerN1> wSum1;
        std::array<float, hLayerN2> wSum2;
        std::array<float, outputN> wSum3;
        std::array<float, hLayerN1> hiddenNeuron1;
        std::array<float, hLayerN2> hiddenNeuron2;
        std::array<float, hLayerN1> deltaHiddenNeuron1;
        std::array<float, hLayerN2> deltaHiddenNeuron2;
        std::array<float, hLayerN1> deltaBias1;
        std::array<float, hLayerN2> deltaBias2;
        std::array<float, outputN> deltaBias3;
        std::array<std::array<float, inputLayerSize>, hLayerN1> deltaWeights1;
        std::array<std::array<float, hLayerN1>, hLayerN2> deltaWeights2;
        std::array<std::array<float, hLayerN2>, outputN> deltaWeights3;
        float costFunction = 0;
        float costFunctionAv = 0;
        float costFunctionTracker = 1;
        float guessProb = 0;
        float guessProbAv = 0;
        float pPrev = 0;
        float pMax = .3;
        int iterations = 1;
        float backPropTerm = 0;
        int loopCounter = 0;
        int targetNumber = 0;
        int epochLength = 100;
        int iterLimit = 100;

        // Open input streams
        mnistParser::training::trainImgStrm.open(images, std::ios::binary);
        mnistParser::training::trainLabelStrm.open(labels, std::ios::binary);
        if (!mnistParser::training::trainImgStrm.is_open() || 
            !mnistParser::training::trainLabelStrm.is_open())
        {
            return;   
        }

        // Neural network loop
        std::cout << "Starting training...\n";
        while (true) {
            loopCounter = 0;
            costFunctionAv = 0;

            while (loopCounter < epochLength) {
                costFunction = 0;

                // Reset
                for (int i = 0; i < targetResult.size(); i++) {
                    targetResult[i] = 0;
                }

                // Get random image
                int imageInd = statpack::randomInt(0, 59999); // min = 0, max = 59999
                std::array<float, mnistParser::IMAGE_PIXELS> inputs = mnistParser::training::getImage(imageInd);
                for (int i = 0; i < inputLayerSize; ++i) {
                    if (inputs[i] > 90)
                        inputs[i] = 255;
                    else
                        inputs[i] = 0;
                }
                int targetNumber = mnistParser::training::getImageNr(imageInd);
                targetResult[targetNumber] = 1;

                // FORWARD PROPAGATION
                inputs = statpack::standardize<inputLayerSize>(inputs);

                // Calculate wsum for each hLayerN1 neurons
                for (int i = 0; i < hLayerN1; i++) {
                    wSum1[i] = statpack::weightedSum<inputLayerSize>(inputs, weights1[i]) + bias1[i];
                    hiddenNeuron1[i] = statpack::sigmoid(wSum1[i]);
                }

                // Calculate wsum for each hLayerN2 neurons
                for (int i = 0; i < hLayerN2; i++) {
                    wSum2[i] = statpack::weightedSum<hLayerN1>(hiddenNeuron1, weights2[i])+bias2[i];
                    hiddenNeuron2[i] = statpack::sigmoid(wSum2[i]);
                }

                // Calculate wsum from the hidden layer to get the final result and then calculate error
                for (int i = 0; i < outputN; i++) {
                    wSum3[i] = statpack::weightedSum<hLayerN2>(hiddenNeuron2, weights3[i])+bias3[i];
                    result[i] = statpack::sigmoid(wSum3[i]);

                    costFunction += std::pow(targetResult[i] - result[i], 2);
                }
                costFunctionAv += costFunction / (float)epochLength;

                // START BACKPROPAGATION
                for (int i = 0; i < outputN; i++) {
                    backPropTerm = -statpack::sigmoidDerivative(wSum3[i])*2*(targetResult[i]-result[i]);

                    for (int k = 0; k < hLayerN2; k++) {
                        deltaWeights3[i][k] += hiddenNeuron2[k] * backPropTerm / (float)epochLength;
                        deltaHiddenNeuron2[k] += weights3[i][k] * backPropTerm;
                    }
                    deltaBias3[i] += backPropTerm / (float)epochLength;
                }

                for (int i = 0; i < hLayerN2; i++) {
                    backPropTerm = statpack::sigmoidDerivative(wSum2[i])*deltaHiddenNeuron2[i];
                    for (int k = 0; k < hLayerN1; k++) {
                        deltaWeights2[i][k] += hiddenNeuron1[k] * backPropTerm / (float)epochLength;
                        deltaHiddenNeuron1[k] += weights2[i][k] * backPropTerm;
                    }
                    deltaBias2[i] += backPropTerm / (float)epochLength;
                    deltaHiddenNeuron2[i] = 0;
                }

                for (int i = 0; i < hLayerN1; i++) {
                    backPropTerm = statpack::sigmoidDerivative(wSum1[i])*deltaHiddenNeuron1[i];
                    for (int k = 0; k < inputLayerSize; k++) {
                        deltaWeights1[i][k] += inputs[k] * backPropTerm / (float)epochLength;
                    }
                    deltaBias1[i] += backPropTerm / (float)epochLength;
                    deltaHiddenNeuron1[i] = 0;
                }

                if (statpack::maxValInd<float, outputN>(result) == targetNumber) {
                    guessProb += 1;
                }

                ++loopCounter;
            }

            ++iterations;

            if (iterations % iterLimit == 0) {
                const float p = guessProb / (float)(epochLength * iterLimit);
                pPrev = p;
                guessProb = 0;

                std::cout << "Guess probability: " << p << " - Cost function: " << costFunctionAv << " - Iterations: " << iterations << "\n";
            } else if (iterations % 20 == 0) {
                std::cout << "...\n";
            }

            if (iterations % 500 == 0) {
                if (costFunctionTracker - costFunctionAv < 0.005 && learnRate <= 250) {
                    learnRate *= 1.25;
                    std::cout << "Seems like a local minima, increasing learn rate to: " << learnRate << "\n";
                } else {
                    costFunctionTracker = costFunctionAv;
                    learnRate *= .5;
                    std::cout << "Learn rate changed to: " << learnRate << "\n";
                    
                    // Save the weights
                    if (true) {
                        std::cout << "Saving new weights and biases...\n";
                        pMax = pPrev;

                        mnistParser::training::outStream.open("./weights1.txt");
                        for (int i = 0; i < weights1[0].size(); ++i) {
                            for (int k = 0; k < weights1.size(); ++k) {
                                mnistParser::training::outStream << weights1[k][i] << "\n";
                            }
                        }
                        mnistParser::training::outStream.close();
                        mnistParser::training::outStream.open("./weights2.txt");
                        for (int i = 0; i < weights2[0].size(); ++i) {
                            for (int k = 0; k < weights2.size(); ++k) {
                                mnistParser::training::outStream << weights2[k][i] << "\n";
                            }
                        }
                        mnistParser::training::outStream.close();
                        mnistParser::training::outStream.open("./weights3.txt");
                        for (int i = 0; i < weights3[0].size(); ++i) {
                            for (int k = 0; k < weights3.size(); ++k) {
                                mnistParser::training::outStream << weights3[k][i] << "\n";
                            }
                        }
                        mnistParser::training::outStream.close();
                        mnistParser::training::outStream.open("./biases1.txt");
                        for (int k = 0; k < bias1.size(); ++k) {
                            mnistParser::training::outStream << bias1[k] << "\n";
                        }
                        mnistParser::training::outStream.close();
                        mnistParser::training::outStream.open("./biases2.txt");
                        for (int k = 0; k < bias2.size(); ++k) {
                            mnistParser::training::outStream << bias2[k] << "\n";
                        }
                        mnistParser::training::outStream.close();
                        mnistParser::training::outStream.open("./biases3.txt");
                        for (int k = 0; k < bias3.size(); ++k) {
                            mnistParser::training::outStream << bias3[k] << "\n";
                        }
                        mnistParser::training::outStream.close();

                        mnistParser::training::outStream.open("./learnrate.txt");
                        mnistParser::training::outStream << learnRate << "\n";
                        mnistParser::training::outStream.close();

                        std::cout << "Weights, biases & learn rate save succesfully!\n";
                    }
                }
            }

            for (int i = 0; i < outputN; i++) {
                for (int k = 0; k < hLayerN2; k++) {
                    weights3[i][k] += -(deltaWeights3[i][k])*learnRate/(float)epochLength;
                    deltaWeights3[i][k] = 0;
                }
                bias3[i] += -deltaBias3[i]*learnRate/(float)epochLength;
                deltaBias3[i] = 0;
            }

            for (int i = 0; i < hLayerN2; i++) {
                for (int k = 0; k < hLayerN1; k++) {
                    weights2[i][k] += -(deltaWeights2[i][k])*learnRate/(float)epochLength;
                    deltaWeights2[i][k] = 0;
                }
                bias2[i] += -deltaBias2[i]*learnRate/(float)epochLength;
                deltaBias2[i] = 0;
            }

            for (int i = 0; i < hLayerN1; i++) {
                for (int k = 0; k < inputLayerSize; k++) {
                    weights1[i][k] += -(deltaWeights1[i][k])*learnRate/(float)epochLength;
                    deltaWeights1[i][k] = 0;
                }
                bias1[i] += -deltaBias1[i]*learnRate/(float)epochLength;
                deltaBias1[i] = 0;
            }
        }

        mnistParser::training::trainImgStrm.close();
        mnistParser::training::trainLabelStrm.close();
    }
}