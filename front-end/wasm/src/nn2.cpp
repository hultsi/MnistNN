#include <iostream>
#include <array>
#include <string>
#include "emscripten.h"

#include "mnistNN.h"
#include "statpack.h"
#include "NeuralNet.h"

constexpr const int INPUT_LAYER_SIZE =    784;
constexpr const int HIDDEN_LAYER_1_SIZE = 40;
constexpr const int HIDDEN_LAYER_2_SIZE = 10;
constexpr const int OUTPUT_LAYER_SIZE = 10;

extern "C" {
    // 28x28 pixel image in
    EMSCRIPTEN_KEEPALIVE void guess_number(int* arrIn, int sizeIn, float* arrOut, int sizeOut) {
        mnistNN::NeuralNet<float, INPUT_LAYER_SIZE, HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE, OUTPUT_LAYER_SIZE> nn("nn1");
        nn.loadWeights("params");
        nn.loadBiases("params");
        
        std::array<float, OUTPUT_LAYER_SIZE> result;
         
        for (int i = 0; i < sizeIn; ++i) {
            nn.inputs[i] = (float) arrIn[i];
        }

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

        for (int i = 0; i < sizeOut; ++i) {
            arrOut[i] = nn.result[i];
        }
    }
}