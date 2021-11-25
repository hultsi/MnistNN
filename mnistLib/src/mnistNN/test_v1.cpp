#include <iostream>
#include <array>
#include <string>

#include "mnistNN.h"
#include "statpack.h"
#include "mnistParser.h"
#include "NeuralNet.h"

// TODO: Change all std::array<std::array<...>> to one dimensionals...
namespace mnistNN {
    void test_v1(std::string images, std::string labels, std::string initValRoot) {
        std::cout << "Initiating test_v1\n";

        constexpr const int INPUT_LAYER_SIZE =    784;
        constexpr const int HIDDEN_LAYER_1_SIZE = 40;
        constexpr const int HIDDEN_LAYER_2_SIZE = 10;
        constexpr const int OUTPUT_LAYER_SIZE = 10;

        NeuralNet<float, INPUT_LAYER_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, OUTPUT_LAYER_SIZE> nn("nn1");
        nn.loadWeights(initValRoot);
        nn.loadBiases(initValRoot);

        std::array<float, OUTPUT_LAYER_SIZE> targetResult;
        std::array<int, OUTPUT_LAYER_SIZE> missed;

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

            nn.inputs = mnistParser::test::getImage(imageInd);
            int targetNumber = mnistParser::test::getImageNr(imageInd);
            targetResult[targetNumber] = 1;

            // Crop & scale image and do preprocessing
            statpack::ImageVector<float> cropped = statpack::cropBlackBackground<float, 28, 28>(nn.inputs);
            nn.inputs = statpack::rescaleImage<float, 28, 28>(cropped.data, cropped.width, cropped.height);
            
            constexpr const int someLimit = 150;
            nn.makeBlackAndWhite(someLimit);
            nn.inputs = statpack::normalize<INPUT_LAYER_SIZE>(nn.inputs, 0, 255, -.5, .5);

            // Calculate wsum for each hLayerN1 neurons
            nn.inputLayerForward();
            nn.hiddenLayer1Forward();
            nn.hiddenLayer2Forward();

            for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
                costFunction += std::pow(targetResult[i] - nn.result[i], 2);
            }
            costFunctionAv += costFunction / static_cast<float>(mnistParser::TEST_IMAGE_MAX);

            if (statpack::maxValInd<float, OUTPUT_LAYER_SIZE>(nn.result) == targetNumber) {
                guessProb += 1;
            } else {
                ++missed[targetNumber];
            }
        }
        const float p = guessProb / static_cast<float>(mnistParser::TEST_IMAGE_MAX);
        std::cout << "-----------------------------------------------------------\n";
        std::cout << "Guess probability: " << p << " - Cost function: " << costFunctionAv << "\n";
        std::cout << "-----------------------------------------------------------\n";

        int sum = 0;
        for (int i = 0; i < 10; ++i) {
            std::cout << "Missed nr " << i << " " << missed[i] << " times\n";
            sum += missed[i];
        }
        std::cout << "Missed total " << sum << "\n";

        mnistParser::test::testImgStrm.close();
        mnistParser::test::testLabelStrm.close();
    }
}