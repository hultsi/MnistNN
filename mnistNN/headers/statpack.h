#pragma once

#include <array>
#include <random>
#include <cmath>

namespace statpack {
    template <int T>
    std::array<float, T> standardize(std::array<float, T> &inputs);
    template <int T>
    std::array<float, T> normalize(std::array<float, T> &inputs, float min0, float max0, float min1, float max1);
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
    template<typename K, int T>
    std::array<K, T> rescaleMnistToHalf(std::array<K, 784> &arr);
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

    template <int T>
    std::array<float, T> normalize(std::array<float, T> &inputs, float min0, float max0, float min1, float max1) {
        std::array<float, T> y;
        for (int i = 0; i < inputs.size(); i++) {
            y[i] = (max1 - min1)/(max0 - min0)*(inputs[i] - max0) + max0;
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

    template<typename K, int T>
    std::array<K, T> rescaleMnistToHalf(std::array<K, 784> &arr) {
        std::array<K, T> out;
        int ind1 = 0;
        int ind2 = 0;
        while (ind2 < T) {
            out[ind2] = (arr[ind1] + arr[ind1+1] + arr[ind1+28] + arr[ind1+29]) / 4;
            if ((ind1 + 2) % 28 == 0) {
                ind1 += 30;
            } else {
                ind1 += 2;
            }  
            ++ind2;
        }
        return out;
    }
}