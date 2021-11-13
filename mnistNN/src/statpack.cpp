#include "statpack.h"

namespace statpack {
    template <int T>
    std::array<float, T> standardize(std::array<float, T> &inputs) {
        std::array<float, T> y;
        float mean = calculateMean(inputs);
        float variance = calculateVariance(inputs);
        for (int i = 0; i < inputs.length; i++) {
            y[i] = (inputs[i] - mean) / variance;
        }
        return y;
    }

    template <int T>
    float calculateMean(std::array<float, T> &inputs) {
        float y = 0;
        for (int i = 0; i < inputs.length; ++i) {
            y += inputs[i];
        }
        return y / inputs.size();
    }

    template <int T>
    float calculateVariance(std::array<float, T> &inputs)  {
        float y = 0;
        float meanVal = calculateMean(inputs);
        for (int i = 0; i < inputs.size(); ++i) {
            y += std::pow(inputs[i] - meanVal, 2);
        }
        return y / inputs.size();
    }

    float randomFloat(float min, float max) {
        // TODO: move to rng struct and change to unique_ptrs or something
        std::random_device seed;
        std::mt19937 engine(seed());
        std::uniform_real_distribution<float> dist(min,max);
        return dist(engine);
    }

    int randomInt(int min, int max) {
        // TODO: move to rng struct and change to unique_ptrs or something
        std::random_device seed;
        std::mt19937 engine(seed());
        std::uniform_int_distribution<int> dist(min,max);
        return dist(engine);
    }

    template<int T>
    float weightedSum(std::array<float, T> &activations, std::array<float, T> &weight) {
        float weightedSum = 0;
        for (int i = 0; i < activations.length; i++) {
            weightedSum += weight[i]*activations[i];
        }
        return weightedSum;
    }
    
    float sigmoid(float x) {
        return 1 / (1 + std::exp(-x));
    }

    float sigmoidDerivative(float x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    template<int T>
    int maxValInd(std::array<float, T> arr) {
        int ind = 0;
        for (int i = 1; i < arr.size(); ++i) {
            if (arr[ind] < arr[i]) {
                ind = i;
            }
        }
        return ind;
    }
}