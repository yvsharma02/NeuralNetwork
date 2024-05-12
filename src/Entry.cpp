#include <iostream>

#include "neural_network/matrix.h"
#include "neural_network/neural_network.h"
#include "reader/data_reader.h"

using namespace NeuralNetwork;

int main() {
    auto x = mnist::read_dataset();

    std::vector<std::pair<Matrix, Matrix>> training_data;
    std::vector<std::pair<Matrix, Matrix>> testing_data;

    for (int i = 0; i < 60000; i++) {
        Matrix ip = Matrix(784, 1);
        Matrix op = Matrix(10, 1);
        for (int j = 0; j < x.training_images[i].size(); j++) {
            ip.value(j, 0) = (real_nnt) x.training_images[i][j] / 255;
        }
        op.value(x.training_labels[i], 0) = 1;
       // op.print();
        training_data.push_back(std::pair<Matrix, Matrix>(std::move(ip), std::move(op)));
    }

    for (int i = 0; i < 10000; i++) {
        Matrix ip = Matrix(784, 1);
        Matrix op = Matrix(10, 1);
        for (int j = 0; j < x.test_images[i].size(); j++) {
            ip.value(j, 0) = (real_nnt)x.test_images[i][j] / 255;
        }
        op.value(x.test_labels[i], 0) = 1;
//        op.print();
        testing_data.push_back(std::pair<Matrix, Matrix>(std::move(ip), std::move(op)));
    }
    Network nn(std::vector<size_nnt>({784, 20, 15, 10}), std::move(training_data), std::move(testing_data));
    nn.train(15, 25);
    nn.test();
    
    return 0;
}