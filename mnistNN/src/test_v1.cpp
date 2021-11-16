#include <iostream>
#include <array>
#include <string>

#include "mnistNN.h"
#include "statpack.h"
#include "mnistParser.h"

// TODO: Change all std::array<std::array<...>> to one dimensionals...
namespace mnistNN {
    void test_v1(std::string images, std::string labels, std::string initValRoot) {
        std::cout << "Initiating test_v1\n";

        constexpr const int inputLayerSize = 784;
        constexpr const int hLayerN1 = 10;
        constexpr const int hLayerN2 = 10;
        constexpr const int outputN = 10;

        std::array<std::array<float, inputLayerSize>, hLayerN1> weights1; //28^2
        std::array<std::array<float, hLayerN1>, hLayerN2> weights2;
        std::array<std::array<float, hLayerN2>, outputN> weights3;
        std::array<float, hLayerN1> bias1;
        std::array<float, hLayerN2> bias2;
        std::array<float, outputN> bias3;

        // Load initial weights
        initWeight<inputLayerSize, hLayerN1>(weights1, initValRoot + "/weights1.txt");
        initWeight<hLayerN1, hLayerN2>(weights2, initValRoot + "/weights2.txt");
        initWeight<hLayerN2, outputN>(weights3, initValRoot + "/weights3.txt");
        initBias<hLayerN1>(bias1, initValRoot + "/biases1.txt");
        initBias<hLayerN2>(bias2, initValRoot + "/biases2.txt");
        initBias<outputN>(bias3, initValRoot + "/biases3.txt");
        
        std::array<float, outputN> result;
        std::array<float, outputN> targetResult;
        std::array<float, hLayerN1> wSum1;
        std::array<float, hLayerN2> wSum2;
        std::array<float, outputN> wSum3;
        std::array<float, hLayerN1> hiddenNeuron1;
        std::array<float, hLayerN2> hiddenNeuron2;

        float costFunction = 0;
        float costFunctionAv = 0;
        float guessProb = 0;
        int targetNumber = 0;

        // Open input streams
        mnistParser::test::testImgStrm.open(images, std::ios::binary);
        mnistParser::test::testLabelStrm.open(labels, std::ios::binary);
        if (!mnistParser::test::testImgStrm.is_open() || 
            !mnistParser::test::testLabelStrm.is_open())
        {
            return;   
        }

        // Neural network loop
        std::cout << "Starting testing...\n";
        for (int imageInd = 0; imageInd < mnistParser::TEST_IMAGE_MAX; ++imageInd) {
            costFunctionAv = 0;
            costFunction = 0;

            // Reset
            for (int i = 0; i < targetResult.size(); i++) {
                targetResult[i] = 0;
            }

            std::array<float, mnistParser::IMAGE_PIXELS> inputs = mnistParser::test::getImage(imageInd);
            for (int i = 0; i < inputLayerSize; ++i) {
                if (inputs[i] > 90)
                    inputs[i] = 255;
                else
                    inputs[i] = 0;
            }
            int targetNumber = mnistParser::test::getImageNr(imageInd);
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
            costFunctionAv += costFunction / (float) mnistParser::TEST_IMAGE_MAX;

            if (statpack::maxValInd<float, outputN>(result) == targetNumber) {
                guessProb += 1;
            }
        }
        const float p = guessProb / (float) mnistParser::TEST_IMAGE_MAX;
        std::cout << "-----------------------------------------------------------\n";
        std::cout << "Guess probability: " << p << " - Cost function: " << costFunctionAv << "\n";
        std::cout << "-----------------------------------------------------------\n";

        mnistParser::test::testImgStrm.close();
        mnistParser::test::testLabelStrm.close();
    }
}