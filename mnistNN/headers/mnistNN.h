#pragma once

#include <array>
#include <string>
#include <fstream>

namespace mnistNN {
    template <int A, int B> 
    void initWeight(std::array<std::array<float, A>, B> &weightArr, std::string filePath);
    template <int A> 
    void initBias(std::array<float, A> &biasArr, std::string filePath);


    void train_v1(std::string images, std::string labels, std::string initValRoot);
}

namespace mnistNN {
    template <int A, int B> 
    void initWeight(std::array<std::array<float, A>, B> &weightArr, std::string filePath) {
        std::ifstream input(filePath);
        float val;
        for (int i = 0; i < weightArr[0].size(); ++i) {
            for (int k = 0; k < weightArr.size(); ++k) {
                if (input >> val)
                    weightArr[k][i] = val;
            }
        }
        input.close();
    }
    
    template <int A> 
    void initBias(std::array<float, A> &biasArr, std::string filePath) {
        std::ifstream input(filePath);
        float val;
        for (int k = 0; k < biasArr.size(); ++k) {
            if (input >> val)
                biasArr[k] = val;
        }
        input.close();
    }
}