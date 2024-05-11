#include <iostream>

#include "neural_network/matrix.h"

int main() {
    NeuralNetwork::Matrix a(2, 3);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            a.value(i, j) = i + j;
        }
    }
//    a.subtract(a);
    a.add(a);
    a.print();
    return 0;
}