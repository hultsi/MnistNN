#include "mnistParser.h"
#include <cstddef> // size_t
#include <fstream>
#include <iostream>

namespace mnistParser {
    int flipInt32(int32_t i) {
        uint8_t a,b,c,d;
        a = i & 255;
        b = (i >> 8) & 255;
        c = (i >> 16) & 255;
        d = (i >> 24) & 255;
        return ((int32_t)a << 24) + ((int32_t)b << 16) + ((int32_t)c << 8) + ((int32_t)d);
    }

    float initLearnRate(std::string filePath) {
        std::ifstream input(filePath);
        float val;
        if (input.is_open()) {
            input >> val;   
        }
        input.close();
        return val;
    }
    
    namespace test {
        std::array<int32_t, IMAGE_PIXELS> currentImageInt;
        std::array<float, IMAGE_PIXELS> currentImageFloat;
        
        int32_t currentImageNr;
        
        std::ifstream testImgStrm;
        std::ifstream testLabelStrm;
        
        std::ofstream outStream;

        // TODO: Optimize some day if I care enough
        // Returns float array for easier life afterwards
        std::array<float, IMAGE_PIXELS> getImage(int32_t nr) {
            int32_t pos = IMAGE_OFFSET + IMAGE_PIXELS * nr;
            if (pos >= TEST_DATA_SIZE) {
                return currentImageFloat;
            }
            if (testImgStrm.is_open()) {
                testImgStrm.seekg(pos, std::ios_base::beg);
                for (int i = 0; i < IMAGE_PIXELS; ++i) {
                    testImgStrm.read(reinterpret_cast<char*>(&currentImageInt[i]), 1);
                    currentImageFloat[i] = (float)currentImageInt[i];
                }
            } else {
                std::cout << "Testing image stream is not open!\n";
            }
            return currentImageFloat;
        }

        int32_t getImageNr(int32_t nr) {
            int32_t pos = LABEL_OFFSET + nr;
            if (pos >= TEST_LABEL_SIZE) {
                return currentImageNr;
            }
            if (testLabelStrm.is_open()) {
                testLabelStrm.seekg(pos, std::ios_base::beg);
                testLabelStrm.read(reinterpret_cast<char*>(&currentImageNr), 1);
            } else {
                std::cout << "Testing label stream is not open!\n";
            }
            return currentImageNr;
        }
    }

    namespace training {
        std::array<int32_t, IMAGE_PIXELS> currentImageInt;
        std::array<float, IMAGE_PIXELS> currentImageFloat;
        
        int32_t currentImageNr;
        
        std::ifstream trainImgStrm;
        std::ifstream trainLabelStrm;
        
        std::ofstream outStream;

        // TODO: Optimize some day if I care enough
        // Returns float array for easier life afterwards
        std::array<float, IMAGE_PIXELS> getImage(int32_t nr) {
            int32_t pos = IMAGE_OFFSET + IMAGE_PIXELS * nr;
            if (pos >= TRAIN_DATA_SIZE) {
                return currentImageFloat;
            }
            if (trainImgStrm.is_open()) {
                trainImgStrm.seekg(pos, std::ios_base::beg);
                for (int i = 0; i < IMAGE_PIXELS; ++i) {
                    trainImgStrm.read(reinterpret_cast<char*>(&currentImageInt[i]), 1);
                    currentImageFloat[i] = (float)currentImageInt[i];
                }
            } else {
                std::cout << "Training image stream is not open!\n";
            }
            return currentImageFloat;
        }

        int32_t getImageNr(int32_t nr) {
            int32_t pos = LABEL_OFFSET + nr;
            if (pos >= TRAIN_LABEL_SIZE) {
                return currentImageNr;
            }
            if (trainLabelStrm.is_open()) {
                trainLabelStrm.seekg(pos, std::ios_base::beg);
                trainLabelStrm.read(reinterpret_cast<char*>(&currentImageNr), 1);
            } else {
                std::cout << "Training label stream is not open!\n";
            }
            return currentImageNr;
        }
    }
}