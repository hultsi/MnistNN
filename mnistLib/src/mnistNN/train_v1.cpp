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
        float learnRate = 100;
        constexpr const int INPUT_LAYER_SIZE =    784;
        constexpr const int HIDDEN_LAYER_1_SIZE = 40;
        constexpr const int HIDDEN_LAYER_2_SIZE = 10;
        constexpr const int OUTPUT_LAYER_SIZE = 10;

        NeuralNet<float, INPUT_LAYER_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, OUTPUT_LAYER_SIZE> nn("nn1");

        // Load initial weights
        if (initValRoot == "") {
            std::cout << "Randomizing nn 1 weights and biases\n";
            nn.randomizeWeights();
            nn.randomizeBiases();
        } else {
            nn.loadWeights(initValRoot);
            nn.loadBiases(initValRoot);
        }

        std::array<float, OUTPUT_LAYER_SIZE> result;
        std::array<float, OUTPUT_LAYER_SIZE> targetResult;

        constexpr const int iterLimit = 100;

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

        // Used to measure drivative of the NN accuracy
        constexpr const int D_LEN = 1000;
        time_point t0;
        time_point t1;
        float p0 = 0.1;
        float p1 = 0.1;

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

                int imageInd = statpack::randomInt(0, 59999); // min = 0, max = 59999
                
                nn.inputs = mnistParser::training::getImage(imageInd);
                int targetNumber = mnistParser::training::getImageNr(imageInd);
                targetResult[targetNumber] = 1;
                
                // Crop & scale image and do preprocessing
                statpack::ImageVector<float> cropped = statpack::cropBlackBackground<float, 28, 28>(nn.inputs);
                nn.inputs = statpack::rescaleImage<float, 28, 28>(cropped.data, cropped.width, cropped.height);
                
                constexpr const int someLimit = 150;
                nn.makeBlackAndWhite(someLimit);
                nn.inputs = statpack::normalize<INPUT_LAYER_SIZE>(nn.inputs, 0, 255, -.5, .5);
                
                // ------------------- //
                // FORWARD PROPAGATION //
                // ------------------- //
                nn.inputLayerForward();
                nn.hiddenLayer1Forward();
                nn.hiddenLayer2Forward();

                for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
                    costFunction += std::pow(targetResult[i] - nn.result[i], 2);
                }
                costFunctionAv += costFunction / (float) epochLength;
                
                // --------------------- //
                // START BACKPROPAGATION //
                // --------------------- //
                nn.hiddenLayer2Backward(targetResult, epochLength);
                nn.hiddenLayer1Backward(epochLength);
                nn.inputLayerBackward(epochLength);

                if (statpack::maxValInd<float, OUTPUT_LAYER_SIZE>(nn.result) == targetNumber) {
                    ++guessProb;
                    ++p1;
                }

                ++loopCounter;
            }

            nn.applyDelta3(epochLength, learnRate);
            nn.applyDelta2(epochLength, learnRate);
            nn.applyDelta1(epochLength, learnRate);

            ++iterations;

            // Then do some info crap printing
            if (iterations % iterLimit == 0) {
                const float p = guessProb / (float) (epochLength * iterLimit);
                pPrev = p;
                guessProb = 0;

                std::cout << "Guess probability: " << p << " - Cost function: " << costFunctionAv << " - Iterations: " << iterations << "\n";
            }

            if (iterations % 1000 == 0) {
                if (costFunctionTracker - costFunctionAv < 0.005 && learnRate <= 50) {
                    learnRate *= 1.1;
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
                        nn.saveWeights("nn1");
                        nn.saveBiases("nn1");
        
        
                        std::cout << "Weights, biases & learn rate save succesfully!\n";
                    }
                }
            }

            if (iterations % D_LEN == 0) {
                p1 /= static_cast<float>(D_LEN * epochLength);
                t1 = std::chrono::steady_clock::now();
                std::chrono::duration<double> diff_time = t1 - t0;
                float diff_probability = p1 - p0;
                
                std::cout << " ------------------------------------------------------------- \n";
                std::cout << "Time for " << D_LEN << " iterations: " << diff_time.count()         << " seconds \n";
                std::cout << "Change of probability: "    << diff_probability                     << " %\n";
                std::cout << "And time derivative: "      << (diff_probability * 60) / diff_time.count() << " %/min\n";
                std::cout << " ------------------------------------------------------------- \n";

                p0 = p1;
                t0 = std::chrono::steady_clock::now();
            }
        }

        mnistParser::training::trainImgStrm.close();
        mnistParser::training::trainLabelStrm.close();
    }
}