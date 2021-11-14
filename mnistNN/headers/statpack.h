#pragma once

#include <array>
#include <random>
#include <cmath>

namespace statpack {
    template <int T>
    std::array<float, T> standardize(std::array<float, T> &inputs);
    template <int T>
    float calculateMean(std::array<float, T> &inputs);
    template <int T>
    float calculateVariance(std::array<float, T> &inputs);
    float randomFloat(float min, float max);
    int randomInt(int min, int max);
    template<int T>
    float weightedSum(std::array<float, T> &activations, std::array<float, T> &weight);
    float sigmoid(float x);
    float sigmoidDerivative(float x);
    template<typename K, int T>
    int maxValInd(std::array<K, T> &arr);
}

namespace statpack {
    template <int T>
    float calculateMean(std::array<float, T> &inputs) {
        float y = 0;
        for (int i = 0; i < inputs.size(); ++i) {
            y += inputs[i];
        }
        return y / inputs.size();
    }
    
    template <int T>
    float calculateVariance(std::array<float, T> &inputs)  {
        float y = 0;
        float meanVal = calculateMean<T>(inputs);
        for (int i = 0; i < inputs.size(); ++i) {
            y += std::pow(inputs[i] - meanVal, 2);
        }
        return y / inputs.size();
    }

    template <int T>
    std::array<float, T> standardize(std::array<float, T> &inputs) {
        std::array<float, T> y;
        float mean = calculateMean<T>(inputs);
        float variance = calculateVariance<T>(inputs);
        for (int i = 0; i < inputs.size(); i++) {
            y[i] = (inputs[i] - mean) / variance;
        }
        return y;
    }

    template<int T>
    float weightedSum(std::array<float, T> &activations, std::array<float, T> &weight) {
        float weightedSum = 0;
        for (int i = 0; i < activations.size(); i++) {
            weightedSum += weight[i] * activations[i];
        }
        return weightedSum;
    }

    template<typename K, int T>
    int maxValInd(std::array<K, T> &arr) {
        int ind = 0;
        for (int i = 1; i < arr.size(); ++i) {
            if (arr[ind] < arr[i]) {
                ind = i;
            }
        }
        return ind;
    }
}