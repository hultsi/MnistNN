#include "statpack.h"

namespace statpack {
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

    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    float sigmoidDerivative(float x) {
        return sigmoid(x) * (1.0f - sigmoid(x));
    }
}