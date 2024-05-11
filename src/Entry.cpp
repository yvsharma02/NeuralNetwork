#include <iostream>

#include "neural_network/matrix.h"

int main() {
    NeuralNetwork::Matrix a(2, 3);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            a.value(i, j) = i + j;
        }
    }
    auto b = a.transpose();
//    a.print();
//    b.print();
    a.multiply(b).print("Mult");
    return 0;
}