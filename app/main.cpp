// TODO: do better
#include <iostream>
#include <string>
#include "mnistNN.h"

int main(int argc, char *argv[]) {
    std::cout << "Running...\n";
    std::string images = "";
    std::string labels = "";
    std::string root   = "";
    std::string root2  = "";
    if (argc >= 3) {
        images = argv[1];
        labels = argv[2];
        if (argc >= 4)
            root = argv[3];
        if (argc == 5) 
            root2 = argv[4];
        mnistNN::train_v1(images, labels, root);
        // mnistNN::train_v2(images, labels, root);
        // mnistNN::train_combined(images, labels, root, root);

        // mnistNN::test_v1(images, labels, root);
        // mnistNN::test_v2(images, labels, root);
        // mnistNN::test_combined(images, labels, root, root);
    } else {
        std::cout << "First argument should be the mnist image binary file " <<
                     "Second argument should be the mnist label binary file\n";
    }
    std::cout << "Ending...\n";
    return 0;
}
