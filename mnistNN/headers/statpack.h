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
    template<int T>
    int maxValInd(std::array<float, T> &arr);
}