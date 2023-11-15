#include "statpack.h"

namespace statpack {
    float randomFloat(const float min, const float max) {
        std::random_device seed;
        std::mt19937 engine(seed());
        std::uniform_real_distribution<float> dist(min,max);
        return dist(engine);
    }

    int randomInt(const int min, const int max) {
        std::random_device seed;
        std::mt19937 engine(seed());
        std::uniform_int_distribution<int> dist(min,max);
        return dist(engine);
    }

    float sigmoid(const float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    float sigmoidDerivative(const float x) {
        return sigmoid(x) * (1.0f - sigmoid(x));
    }
}