#include <iostream>
#include <array>
#include <string>
#include <chrono>

#include "mnistNN.h"
#include "statpack.h"
#include "mnistParser.h"
#include "NeuralNet.h"

using time_point = std::chrono::time_point<std::chrono::steady_clock>;

// TODO: Change all std::array<std::array<...>> to one dimensionals...
namespace mnistNN {
    void train_combined(std::string images, std::string labels, std::string initValRoot1, std::string initValRoot2) {
        std::cout << "Initiating train_combined\n";

        constexpr const int INPUT_LAYER_SIZE_1 =    784;
        constexpr const int HIDDEN_LAYER_1_SIZE_1 = 20;
        constexpr const int HIDDEN_LAYER_2_SIZE_1 = 10;

        constexpr const int INPUT_LAYER_SIZE_2 =    196;
        constexpr const int HIDDEN_LAYER_1_SIZE_2 = 20;
        constexpr const int HIDDEN_LAYER_2_SIZE_2 = 10;

        constexpr const int OUTPUT_LAYER_SIZE = 10;

        NeuralNet<float, INPUT_LAYER_SIZE_1, HIDDEN_LAYER_1_SIZE_1, HIDDEN_LAYER_2_SIZE_1, OUTPUT_LAYER_SIZE> nn_1("nn1");
        NeuralNet<float, INPUT_LAYER_SIZE_2, HIDDEN_LAYER_1_SIZE_2, HIDDEN_LAYER_2_SIZE_2, OUTPUT_LAYER_SIZE> nn_2("nn2");

        if (initValRoot1 == "") {
            std::cout << "Randomizing nn 1 weights and biases\n";
            nn_1.randomizeWeights();
            nn_1.randomizeBiases();
        } else {
            nn_1.loadWeights(initValRoot1);
            nn_1.loadBiases(initValRoot1);
        }
        if (initValRoot2 == "") {
            std::cout << "Randomizing nn 2 weights and biases\n";
            nn_2.randomizeWeights();
            nn_2.randomizeBiases();
        } else {
            nn_2.loadWeights(initValRoot2);
            nn_2.loadBiases(initValRoot2);
        }

        std::array<float, OUTPUT_LAYER_SIZE> result;
        std::array<float, OUTPUT_LAYER_SIZE> targetResult;
        
        float learnRate = 300;
        float costFunction = 0;
        float costFunctionAv = 0;
        float costFunctionTracker = 1;
        float guessProb = 0;
        float guessProbAv = 0;
        float pPrev = 0;
        float pMax = .15;
        int iterations = 1;
        int loopCounter = 0;
        int targetNumber = 0;
        int epochLength = 100;

        constexpr const int iterLimit = 100;

        // Used to measure drivative of the NN accuracy
        constexpr const int D_LEN = 1000;
        time_point t0;
        time_point t1;
        int p0 = 0.1;
        int p1 = 0.1;

        mnistParser::training::trainImgStrm.open(images, std::ios::binary);
        mnistParser::training::trainLabelStrm.open(labels, std::ios::binary);
        if (!mnistParser::training::trainImgStrm.is_open() || 
            !mnistParser::training::trainLabelStrm.is_open())
        {
            return;   
        }

        std::cout << "Starting training...\n";
        t0 = std::chrono::steady_clock::now();
        while (true) {
            loopCounter = 0;
            costFunctionAv = 0;

            while (loopCounter < epochLength) {
                costFunction = 0;

                // Reset
                for (int i = 0; i < targetResult.size(); i++) {
                    targetResult[i] = 0;
                }

                int imageInd = statpack::randomInt(0, 59999); // min = 0, max = 59999
                
                int targetNumber = mnistParser::training::getImageNr(imageInd);
                targetResult[targetNumber] = 1;

                nn_1.inputs = mnistParser::training::getImage(imageInd);
                nn_2.inputs = statpack::rescaleMnistToHalf<float, INPUT_LAYER_SIZE_1, INPUT_LAYER_SIZE_2>(nn_1.inputs);

                constexpr const int someLimit = 80;
                nn_1.makeBlackAndWhite(someLimit);
                nn_2.makeBlackAndWhite(someLimit);

                nn_1.inputs = statpack::standardize<INPUT_LAYER_SIZE_1>(nn_1.inputs);
                nn_2.inputs = statpack::standardize<INPUT_LAYER_SIZE_2>(nn_2.inputs);

                // ------------------- //
                // FORWARD PROPAGATION //
                // ------------------- //
                nn_1.inputLayerForward();
                nn_2.inputLayerForward();

                nn_1.hiddenLayer1Forward();
                nn_2.hiddenLayer1Forward();

                // TODO: do this better
                // Combine nn1 & nn2 wsum3 by summing and sigmoiding them
                for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
                    nn_1.wSum3[i] = statpack::weightedSum<HIDDEN_LAYER_2_SIZE_1>(nn_1.hiddenNeuron2, nn_1.weights3[i]) + nn_1.bias3[i];
                    nn_2.wSum3[i] = statpack::weightedSum<HIDDEN_LAYER_2_SIZE_2>(nn_2.hiddenNeuron2, nn_2.weights3[i]) + nn_2.bias3[i];
                    result[i] = statpack::sigmoid(nn_1.wSum3[i] + nn_2.wSum3[i]);

                    costFunction += std::pow(targetResult[i] - result[i], 2);
                }
                costFunctionAv += costFunction / (float) epochLength;
                
                // --------------------- //
                // START BACKPROPAGATION //
                // --------------------- //
                for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
                    nn_1.backPropTerm = (-1) * 2 * statpack::sigmoidDerivative(nn_1.wSum3[i] + nn_2.wSum3[i]) * (targetResult[i] - result[i]);
                    for (int k = 0; k < HIDDEN_LAYER_2_SIZE_1; k++) {
                        nn_1.deltaWeights3[i][k] += nn_1.hiddenNeuron2[k] * nn_1.backPropTerm / (float) epochLength;
                        nn_1.deltaHiddenNeuron2[k] += nn_1.weights3[i][k] * nn_1.backPropTerm;
                    }
                    nn_1.deltaBias3[i] += nn_1.backPropTerm / (float) epochLength;

                    nn_2.backPropTerm = (-1) * 2 * statpack::sigmoidDerivative(nn_1.wSum3[i] + nn_2.wSum3[i]) * (targetResult[i] - result[i]);
                    for (int k = 0; k < HIDDEN_LAYER_2_SIZE_2; k++) {
                        nn_2.deltaWeights3[i][k] += nn_2.hiddenNeuron2[k] * nn_2.backPropTerm / (float) epochLength;
                        nn_2.deltaHiddenNeuron2[k] += nn_2.weights3[i][k] * nn_2.backPropTerm;
                    }
                    nn_2.deltaBias3[i] += nn_2.backPropTerm / (float) epochLength;
                }

                nn_1.hiddenLayer1Backward(epochLength);
                nn_2.hiddenLayer1Backward(epochLength);

                nn_1.inputLayerBackward(epochLength);
                nn_2.inputLayerBackward(epochLength);

                if (statpack::maxValInd<float, OUTPUT_LAYER_SIZE>(result) == targetNumber) {
                    ++guessProb;
                    ++p1;
                }

                ++loopCounter;
            }

            nn_1.applyDelta3(epochLength, learnRate);
            nn_2.applyDelta3(epochLength, learnRate);

            nn_1.applyDelta2(epochLength, learnRate);
            nn_2.applyDelta2(epochLength, learnRate);

            nn_1.applyDelta1(epochLength, learnRate);
            nn_2.applyDelta1(epochLength, learnRate);

            ++iterations;

            if (iterations % iterLimit == 0) {
                const float p = guessProb / (float) (epochLength * iterLimit);
                pPrev = p;
                guessProb = 0;

                std::cout << "Guess probability: " << p << " - Cost function: " << costFunctionAv << " - Iterations: " << iterations << "\n";
            }

            if (iterations % 500 == 0) {
                if (costFunctionTracker - costFunctionAv < 0.005 && learnRate <= 250) {
                    learnRate *= 1.25;
                    std::cout << "Seems like a local minima, increasing learn rate to: " << learnRate << "\n";
                } else {
                    costFunctionTracker = costFunctionAv;
                    learnRate *= .5;
                    std::cout << "Learn rate changed to: " << learnRate << "\n";
                    
                    // ---------------- //
                    // Save the weights //
                    // ---------------- //
                    if (pPrev > pMax) {
                        std::cout << "Saving new weights and biases...\n";
                        pMax = pPrev;
                        nn_1.saveWeights("nn1");
                        nn_1.saveBiases("nn1");
                        nn_2.saveWeights("nn2");
                        nn_2.saveBiases("nn2");
                        std::cout << "Weights, biases & learn rate save succesfully!\n";
                    }
                }
            }

            if (iterations % 1000 == 0) {
                t1 = std::chrono::steady_clock::now();
                std::chrono::duration<double> diff_time = t1 - t0;
                float diff_probability = p1 - p0;
                std::cout << "Time for 1000 iterations: " << diff_time.count()                    << "\n" <<
                             "Change of probability: "    << diff_probability                     << "\n" <<
                             "And time derivative: "      << diff_probability / diff_time.count() << "1/s";
                p0 = p1;
                t0 = t1;
            }
        }

        mnistParser::training::trainImgStrm.close();
        mnistParser::training::trainLabelStrm.close();
    }
}