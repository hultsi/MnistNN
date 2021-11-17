#pragma once

#include <iostream>
#include <fstream>

#include <array>
#include <string>

#include "mnistNN.h"
#include "statpack.h"
#include "mnistParser.h"

// TODO: Change all std::array<std::array<...>> to one dimensionals...
namespace mnistNN {

    template <typename T, int A, int B, int C, int D>
    class NeuralNet {
        public:
        static const int inputLayerSize = A;
        static const int hLayerN1 = B;
        static const int hLayerN2 = C;
        static const int outputN = D;

        std::array<float, inputLayerSize> inputs;

        std::array<std::array<T, inputLayerSize>, hLayerN1> weights1{};
        std::array<std::array<T, hLayerN1>, hLayerN2>       weights2{};
        std::array<std::array<T, hLayerN2>, outputN>        weights3{};
        std::array<T, hLayerN1>                             bias1{};
        std::array<T, hLayerN2>                             bias2{};
        std::array<T, outputN>                              bias3{};

        std::array<T, hLayerN1>                             wSum1{};
        std::array<T, hLayerN2>                             wSum2{};
        std::array<T, outputN>                              wSum3{};
        std::array<T, hLayerN1>                             hiddenNeuron1{};
        std::array<T, hLayerN2>                             hiddenNeuron2{};

        std::array<T, hLayerN1>                             deltaHiddenNeuron1{};
        std::array<T, hLayerN2>                             deltaHiddenNeuron2{};
        std::array<std::array<T, inputLayerSize>, hLayerN1> deltaWeights1{};
        std::array<std::array<T, hLayerN1>, hLayerN2>       deltaWeights2{};
        std::array<std::array<T, hLayerN2>, outputN>        deltaWeights3{};
        std::array<T, hLayerN1>                             deltaBias1{};
        std::array<T, hLayerN2>                             deltaBias2{};
        std::array<T, outputN>                              deltaBias3{};

        std::array<T, outputN>                              result{};

        std::ofstream paramStreamOut;
        std::ifstream paramStreamIn;

        float backPropTerm = 0;
        std::string id = "";

        NeuralNet(std::string id) : id(id) {
        }

        void randomizeWeights() {
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
        }

        void randomizeBiases() {
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
        }

        void makeBlackAndWhite(const int limit) {
            for (int i = 0; i < inputLayerSize; ++i) {
                if (inputs[i] > limit)
                    inputs[i] = 255;
                else
                    inputs[i] = 0;
            }
        }

        void inputLayerForward() {
            // Calculate wsum for each hLayerN1 neurons
            for (int i = 0; i < hLayerN1; i++) {
                wSum1[i] = statpack::weightedSum<inputLayerSize>(inputs, weights1[i]) + bias1[i];
                hiddenNeuron1[i] = statpack::sigmoid(wSum1[i]);
            }
        }

        void hiddenLayer1Forward() {
            // Calculate wsum for each hLayerN2 neurons
            for (int i = 0; i < hLayerN2; i++) {
                wSum2[i] = statpack::weightedSum<hLayerN1>(hiddenNeuron1, weights2[i]) + bias2[i];
                hiddenNeuron2[i] = statpack::sigmoid(wSum2[i]);
            }
        }

        void hiddenLayer1Backward(int epochLength) {
            for (int i = 0; i < hLayerN2; i++) {
                backPropTerm = statpack::sigmoidDerivative(wSum2[i]) * deltaHiddenNeuron2[i];
                for (int k = 0; k < hLayerN1; k++) {
                    deltaWeights2[i][k] += hiddenNeuron1[k] * backPropTerm / (float) epochLength;
                    deltaHiddenNeuron1[k] += weights2[i][k] * backPropTerm;
                }
                deltaBias2[i] += backPropTerm / (float) epochLength;
                deltaHiddenNeuron2[i] = 0;
            }
        }

        void inputLayerBackward(int epochLength) {
            for (int i = 0; i < hLayerN1; i++) {
                backPropTerm = statpack::sigmoidDerivative(wSum1[i])*deltaHiddenNeuron1[i];
                for (int k = 0; k < inputLayerSize; k++) {
                    deltaWeights1[i][k] += inputs[k] * backPropTerm / (float) epochLength;
                }
                deltaBias1[i] += backPropTerm / (float) epochLength;
                deltaHiddenNeuron1[i] = 0;
            }
        }

        // Note: factor is generally learnRate/epochLength
        void applyDelta3(int epochLength, float learnRate) {
            for (int i = 0; i < outputN; i++) {
                for (int k = 0; k < hLayerN2; k++) {
                    weights3[i][k] += -(deltaWeights3[i][k]) * learnRate / (float) epochLength;
                    deltaWeights3[i][k] = 0;
                }
                bias3[i] += -deltaBias3[i] * learnRate / (float) epochLength;
                deltaBias3[i] = 0;
            }
        }

        void applyDelta2(int epochLength, float learnRate) {
            for (int i = 0; i < hLayerN2; i++) {
                for (int k = 0; k < hLayerN1; k++) {
                    weights2[i][k] += -(deltaWeights2[i][k]) * learnRate / (float) epochLength;
                    deltaWeights2[i][k] = 0;
                }
                bias2[i] += -deltaBias2[i] * learnRate / (float) epochLength;
                deltaBias2[i] = 0;
            }
        }

        void applyDelta1(int epochLength, float learnRate) {
            for (int i = 0; i < hLayerN1; i++) {
                for (int k = 0; k < inputLayerSize; k++) {
                    weights1[i][k] += -(deltaWeights1[i][k]) * learnRate / (float) epochLength;
                    deltaWeights1[i][k] = 0;
                }
                bias1[i] += -deltaBias1[i] * learnRate / (float) epochLength;
                deltaBias1[i] = 0;
            }
        }

        void saveWeights(std::string id) {
            std::string filename = id + "_weights1.txt";
            paramStreamOut.open("./" + filename);
            for (int i = 0; i < weights1[0].size(); ++i) {
                for (int k = 0; k < weights1.size(); ++k) {
                    paramStreamOut << weights1[k][i] << "\n";
                }
            }
            paramStreamOut.close();
            
            filename = id + "_weights2.txt";
            paramStreamOut.open("./" + filename);
            for (int i = 0; i < weights2[0].size(); ++i) {
                for (int k = 0; k < weights2.size(); ++k) {
                    paramStreamOut << weights2[k][i] << "\n";
                }
            }
            paramStreamOut.close();

            filename = id + "_weights3.txt";
            paramStreamOut.open("./" + filename);
            for (int i = 0; i < weights3[0].size(); ++i) {
                for (int k = 0; k < weights3.size(); ++k) {
                    paramStreamOut << weights3[k][i] << "\n";
                }
            }
            paramStreamOut.close();   
        }

        void saveBiases(std::string id) {
            std::string filename = id + "_biases1.txt";
            paramStreamOut.open("./" + filename);
            for (int k = 0; k < bias1.size(); ++k) {
                paramStreamOut << bias1[k] << "\n";
            }
            paramStreamOut.close();

            filename = id + "_biases2.txt";
            paramStreamOut.open("./" + filename);
            for (int k = 0; k < bias2.size(); ++k) {
                paramStreamOut << bias2[k] << "\n";
            }
            paramStreamOut.close();

            filename = id + "_biases3.txt";
            paramStreamOut.open("./" + filename);
            for (int k = 0; k < bias3.size(); ++k) {
                paramStreamOut << bias3[k] << "\n";
            }
            paramStreamOut.close();
        }

        void loadWeights(std::string filePath) {
            filePath += "/";
            filePath += id;
            std::string file = filePath + "_weights1.txt";
            paramStreamIn.open(file);
            float val;
            for (int i = 0; i < weights1[0].size(); ++i) {
                for (int k = 0; k < weights1.size(); ++k) {
                    if (paramStreamIn >> val)
                        weights1[k][i] = val;
                }
            }
            paramStreamIn.close();

            file = filePath + "_weights2.txt";
            paramStreamIn.open(file);
            for (int i = 0; i < weights2[0].size(); ++i) {
                for (int k = 0; k < weights2.size(); ++k) {
                    if (paramStreamIn >> val)
                        weights2[k][i] = val;
                }
            }
            paramStreamIn.close();

            file = filePath + "_weights3.txt";
            paramStreamIn.open(file);
            for (int i = 0; i < weights3[0].size(); ++i) {
                for (int k = 0; k < weights3.size(); ++k) {
                    if (paramStreamIn >> val)
                        weights3[k][i] = val;
                }
            }
            paramStreamIn.close();
        }

        void loadBiases(std::string filePath) {
            filePath += "/";
            filePath += id;
            std::string file = filePath + "_biases1.txt";
            paramStreamIn.open(file);
            float val;
            for (int i = 0; i < bias1.size(); ++i) {
                if (paramStreamIn >> val)
                    bias1[i] = val;
            }
            paramStreamIn.close();

            file = filePath + "_biases2.txt";
            paramStreamIn.open(file);
            for (int i = 0; i < bias2.size(); ++i) {
                if (paramStreamIn >> val)
                    bias2[i] = val;
            }
            paramStreamIn.close();

            file = filePath + "_biases3.txt";
            paramStreamIn.open(file);
            for (int i = 0; i < bias3.size(); ++i) {
                if (paramStreamIn >> val)
                    bias3[i] = val;
            }
            paramStreamIn.close();
        }
    };

}