#include <iostream>
#include <array>
#include <string>
#include "emscripten.h"

#include "mnistNN.h"
#include "statpack.h"
#include "NeuralNet.h"

constexpr const int INPUT_LAYER_SIZE_1 =    784;
constexpr const int HIDDEN_LAYER_1_SIZE_1 = 10;
constexpr const int HIDDEN_LAYER_2_SIZE_1 = 40;

constexpr const int INPUT_LAYER_SIZE_2 =    196;
constexpr const int HIDDEN_LAYER_1_SIZE_2 = 10;
constexpr const int HIDDEN_LAYER_2_SIZE_2 = 40;

constexpr const int OUTPUT_LAYER_SIZE = 10;

extern "C" {
    // 28x28 pixel image in
    EMSCRIPTEN_KEEPALIVE void guess_number(int* arrIn, int sizeIn, float* arrOut, int sizeOut) {
        mnistNN::NeuralNet<float, INPUT_LAYER_SIZE_1, HIDDEN_LAYER_1_SIZE_1, HIDDEN_LAYER_2_SIZE_1, OUTPUT_LAYER_SIZE> nn_1("nn1");
        mnistNN::NeuralNet<float, INPUT_LAYER_SIZE_2, HIDDEN_LAYER_1_SIZE_2, HIDDEN_LAYER_2_SIZE_2, OUTPUT_LAYER_SIZE> nn_2("nn2");
        
        nn_1.loadWeights("params");
        nn_1.loadBiases("params");
        nn_2.loadWeights("params");
        nn_2.loadBiases("params");
        
        std::array<float, OUTPUT_LAYER_SIZE> result;
        std::array<float, OUTPUT_LAYER_SIZE> targetResult;
        
        std::array<int, OUTPUT_LAYER_SIZE> missed;
         
        for (int i = 0; i < sizeIn; ++i) {
            nn_1.inputs[i] = (float) arrIn[i];
        }
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

        // Combine nn1 & nn2 wsum3 by summing and sigmoiding them
        for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
            nn_1.wSum3[i] = statpack::weightedSum<HIDDEN_LAYER_2_SIZE_1>(nn_1.hiddenNeuron2, nn_1.weights3[i]) + nn_1.bias3[i];
            nn_2.wSum3[i] = statpack::weightedSum<HIDDEN_LAYER_2_SIZE_2>(nn_2.hiddenNeuron2, nn_2.weights3[i]) + nn_2.bias3[i];
            result[i] = statpack::sigmoid( (nn_1.wSum3[i] + nn_2.wSum3[i]) / 2 );
        }

        for (int i = 0; i < sizeOut; ++i) {
            arrOut[i] = result[i];
        }
    }
}