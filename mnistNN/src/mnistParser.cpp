#include "mnistParser.h"
#include <cstddef> // size_t
#include <fstream>

namespace mnistParser {
    int flipInt32(int32_t i) {
        uint8_t a,b,c,d;
        a = i & 255;
        b = (i >> 8) & 255;
        c = (i >> 16) & 255;
        d = (i >> 24) & 255;
        return ((int32_t)a << 24) + ((int32_t)b << 16) + ((int32_t)c << 8) + ((int32_t)d);
    }

    namespace test {

    }

    namespace training {
        std::array<int32_t, IMAGE_PIXELS> currentImageInt;
        std::array<float, IMAGE_PIXELS> currentImageFloat;
        
        int32_t currentImageNr;
        
        std::ifstream trainImgStrm;
        std::ifstream trainLabelStrm;
        
        std::ofstream saveWeights1;
        std::ofstream saveWeights2;
        std::ofstream saveWeights3;
        std::ofstream saveBiases1;
        std::ofstream saveBiases2;
        std::ofstream saveBiases3;

        // TODO: Optimize some day if I care enough
        // Returns float array for easier life afterwards
        std::array<float, IMAGE_PIXELS> getImage(int32_t nr) {
            int32_t pos = IMAGE_OFFSET + IMAGE_PIXELS * nr;
            if (pos >= TRAIN_DATA_SIZE) {
                return currentImageFloat;
            }
            trainImgStrm.seekg(pos, std::ios_base::beg);
            for (int i = 0; i < IMAGE_PIXELS; ++i) {
                trainImgStrm.read(reinterpret_cast<char*>(&currentImageInt[i]), 1);
                currentImageFloat[i] = (float)currentImageInt[i];
            }
            return currentImageFloat;
        }

        int32_t getImageNr(int32_t nr) {
            int32_t pos = LABEL_OFFSET + nr;
            if (pos >= TRAIN_LABEL_SIZE) {
                return currentImageNr;
            }
            trainLabelStrm.seekg(pos, std::ios_base::beg);
            return currentImageNr;
        }
    }
}