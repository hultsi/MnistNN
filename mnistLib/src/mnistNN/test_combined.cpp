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
    void test_combined(std::string images, std::string labels, std::string initValRoot1, std::string initValRoot2) {
        std::cout << "Initiating test_combined\n";

        constexpr const int INPUT_LAYER_SIZE_1 =    784;
        constexpr const int HIDDEN_LAYER_1_SIZE_1 = 10;
        constexpr const int HIDDEN_LAYER_2_SIZE_1 = 40;

        constexpr const int INPUT_LAYER_SIZE_2 =    196;
        constexpr const int HIDDEN_LAYER_1_SIZE_2 = 10;
        constexpr const int HIDDEN_LAYER_2_SIZE_2 = 40;

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
        
        std::array<int, OUTPUT_LAYER_SIZE> missed;
        
        float costFunction = 0;
        float guessProb = 0;
        int targetNumber = 0;

        mnistParser::training::trainImgStrm.open(images, std::ios::binary);
        mnistParser::training::trainLabelStrm.open(labels, std::ios::binary);
        if (!mnistParser::training::trainImgStrm.is_open() || 
            !mnistParser::training::trainLabelStrm.is_open())
        {
            return;   
        }

        std::cout << "Starting testing...\n";
        for (int imageInd = 0; imageInd < 10000; ++imageInd) {
            // Reset
            for (int i = 0; i < targetResult.size(); i++) {
                targetResult[i] = 0;
            }
            
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
                result[i] = statpack::sigmoid( (nn_1.wSum3[i] + nn_2.wSum3[i]) / 2 );

                costFunction += std::pow(targetResult[i] - result[i], 2);
            }
            
            if (statpack::maxValInd<float, OUTPUT_LAYER_SIZE>(result) == targetNumber) {
                ++guessProb;
            } else {
                // std::cout << "Missed " << targetNumber << "\n";
                ++missed[targetNumber];
            }
        }
                
        std::cout << " ------------------------------------------------------------- \n";
        std::cout << "Guess probability: " << guessProb / static_cast<float>(mnistParser::TEST_IMAGE_MAX) << " - Cost function: " << costFunction / static_cast<float>(mnistParser::TEST_IMAGE_MAX) << " seconds \n";
        std::cout << " ------------------------------------------------------------- \n";

        int sum = 0;
        for (int i = 0; i < 10; ++i) {
            std::cout << "Missed nr " << i << " " << missed[i] << " times\n";
            sum += missed[i];
        }
        std::cout << "Missed total " << sum << "\n";

        mnistParser::training::trainImgStrm.close();
        mnistParser::training::trainLabelStrm.close();
    }
}