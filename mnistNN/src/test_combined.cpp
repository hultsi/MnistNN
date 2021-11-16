#include <iostream>
#include <array>
#include <string>

#include "mnistNN.h"
#include "statpack.h"
#include "mnistParser.h"

// TODO: Change all std::array<std::array<...>> to one dimensionals...
namespace mnistNN {
    struct V1 {
        static const int inputLayerSize = 784;
        static const int hLayerN1 = 10;
        static const int hLayerN2 = 10;
        static const int outputN = 10;

        std::array<std::array<float, inputLayerSize>, hLayerN1> weights1; //28^2
        std::array<std::array<float, hLayerN1>, hLayerN2> weights2;
        std::array<std::array<float, hLayerN2>, outputN> weights3;
        std::array<float, hLayerN1> bias1;
        std::array<float, hLayerN2> bias2;
        std::array<float, outputN> bias3;

        std::array<float, outputN> result;
        std::array<float, hLayerN1> wSum1;
        std::array<float, hLayerN2> wSum2;
        std::array<float, outputN> wSum3;
        std::array<float, hLayerN1> hiddenNeuron1;
        std::array<float, hLayerN2> hiddenNeuron2;
    };

    struct V2 {
        static const int inputLayerSize = 196;
        static const int hLayerN1 = 20;
        static const int hLayerN2 = 10;
        static const int outputN = 10;

        std::array<std::array<float, inputLayerSize>, hLayerN1> weights1; //28^2
        std::array<std::array<float, hLayerN1>, hLayerN2> weights2;
        std::array<std::array<float, hLayerN2>, outputN> weights3;
        std::array<float, hLayerN1> bias1;
        std::array<float, hLayerN2> bias2;
        std::array<float, outputN> bias3;
        
        std::array<float, outputN> result;
        std::array<float, hLayerN1> wSum1;
        std::array<float, hLayerN2> wSum2;
        std::array<float, outputN> wSum3;
        std::array<float, hLayerN1> hiddenNeuron1;
        std::array<float, hLayerN2> hiddenNeuron2;
    };

    void test_combined(std::string images, std::string labels, std::string initValRoot_v1, std::string initValRoot_v2) {
        std::cout << "Initiating test_combined\n";

        V1 v1;
        V2 v2;

        // Load initial weights
        initWeight<v1.inputLayerSize, v1.hLayerN1>(v1.weights1, initValRoot_v1 + "/weights1.txt");
        initWeight<v1.hLayerN1, v1.hLayerN2>(v1.weights2, initValRoot_v1 + "/weights2.txt");
        initWeight<v1.hLayerN2, v1.outputN>(v1.weights3, initValRoot_v1 + "/weights3.txt");
        initBias<v1.hLayerN1>(v1.bias1, initValRoot_v1 + "/biases1.txt");
        initBias<v1.hLayerN2>(v1.bias2, initValRoot_v1 + "/biases2.txt");
        initBias<v1.outputN>(v1.bias3, initValRoot_v1 + "/biases3.txt");
        
        // Load initial weights
        initWeight<v2.inputLayerSize, v2.hLayerN1>(v2.weights1, initValRoot_v2 + "/weights1.txt");
        initWeight<v2.hLayerN1, v2.hLayerN2>(v2.weights2, initValRoot_v2 + "/weights2.txt");
        initWeight<v2.hLayerN2, v2.outputN>(v2.weights3, initValRoot_v2 + "/weights3.txt");
        initBias<v2.hLayerN1>(v2.bias1, initValRoot_v2 + "/biases1.txt");
        initBias<v2.hLayerN2>(v2.bias2, initValRoot_v2 + "/biases2.txt");
        initBias<v2.outputN>(v2.bias3, initValRoot_v2 + "/biases3.txt");

        std::array<float, 10> result;
        std::array<float, 10> targetResult;

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
            std::array<float, v2.inputLayerSize> inputsv2 = statpack::rescaleMnistToHalf<float, v2.inputLayerSize>(inputs);
            int targetNumber = mnistParser::test::getImageNr(imageInd);
            targetResult[targetNumber] = 1;


            // Calculate version 1 result
            for (int i = 0; i < v1.inputLayerSize; ++i) {
                if (inputs[i] > 90)
                    inputs[i] = 255;
                else
                    inputs[i] = 0;
            }
            inputs = statpack::standardize<v1.inputLayerSize>(inputs);
            // Calculate wsum for each hLayerN1 neurons
            for (int i = 0; i < v1.hLayerN1; i++) {
                v1.wSum1[i] = statpack::weightedSum<v1.inputLayerSize>(inputs, v1.weights1[i]) + v1.bias1[i];
                v1.hiddenNeuron1[i] = statpack::sigmoid(v1.wSum1[i]);
            }

            // Calculate wsum for each hLayerN2 neurons
            for (int i = 0; i < v1.hLayerN2; i++) {
                v1.wSum2[i] = statpack::weightedSum<v1.hLayerN1>(v1.hiddenNeuron1, v1.weights2[i]) + v1.bias2[i];
                v1.hiddenNeuron2[i] = statpack::sigmoid(v1.wSum2[i]);
            }

            // Calculate wsum from the hidden layer to get the final result and then calculate error
            for (int i = 0; i < v1.outputN; i++) {
                v1.wSum3[i] = statpack::weightedSum<v1.hLayerN2>(v1.hiddenNeuron2, v1.weights3[i]) + v1.bias3[i];
                v1.result[i] = statpack::sigmoid(v1.wSum3[i]);
            }

            // Calculate version 2 result
            for (int i = 0; i < v2.inputLayerSize; ++i) {
                if (inputsv2[i] > 90)
                    inputsv2[i] = 255;
                else
                    inputsv2[i] = 0;
            }
            inputsv2 = statpack::standardize<v2.inputLayerSize>(inputsv2);
            // Calculate wsum for each hLayerN1 neurons
            for (int i = 0; i < v2.hLayerN1; i++) {
                v2.wSum1[i] = statpack::weightedSum<v2.inputLayerSize>(inputsv2, v2.weights1[i]) + v2.bias1[i];
                v2.hiddenNeuron1[i] = statpack::sigmoid(v2.wSum1[i]);
            }

            // Calculate wsum for each hLayerN2 neurons
            for (int i = 0; i < v2.hLayerN2; i++) {
                v2.wSum2[i] = statpack::weightedSum<v2.hLayerN1>(v2.hiddenNeuron1, v2.weights2[i]) + v2.bias2[i];
                v2.hiddenNeuron2[i] = statpack::sigmoid(v2.wSum2[i]);
            }

            // Calculate wsum from the hidden layer to get the final result and then calculate error
            for (int i = 0; i < v2.outputN; i++) {
                v2.wSum3[i] = statpack::weightedSum<v2.hLayerN2>(v2.hiddenNeuron2, v2.weights3[i]) + v2.bias3[i];
                v2.result[i] = statpack::sigmoid(v2.wSum3[i]);
            }

            for (int i = 0; i < 10; ++i) {
                result[i] = (v1.result[i] + v2.result[i]) / 2;
                costFunction += std::pow(targetResult[i] - result[i], 2);
                costFunctionAv += costFunction / (float) mnistParser::TEST_IMAGE_MAX;
            }

            if (statpack::maxValInd<float, 10>(result) == targetNumber) {
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