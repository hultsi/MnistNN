// THIS FILE IS ATM USED FOR TESTING
// THAT EVERYTHING WORKS AS INTENDED
#include <iostream>
#include <string>
#include "mnistNN.h"

int main(int argc, char *argv[]) {
    std::cout << "Running...\n";
    std::string images;
    std::string labels;
    if (argc > 1) {
        images = argv[1];
        labels = argv[2];
    }
    mnistNN::train_v1(images, labels);
    std::cout << "Ending...\n";
    return 0;
}
