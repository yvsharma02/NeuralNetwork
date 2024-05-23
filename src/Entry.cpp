#include <iostream>

#include "neural_network/matrix.h"
#include "neural_network/neural_network.h"
#include "reader/data_reader.h"
#include <cl/cl.h>
#include <stdlib.h>

int main() {
    srand(1234);
    auto x = mnist::read_dataset();

    std::vector<std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>> training_data;
    std::vector<std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>> testing_data;
    
    for (int i = 0; i < 64 * 937; i++) {
        NeuralNetwork::Matrix ip = NeuralNetwork::Matrix(784, 1);
        NeuralNetwork::Matrix op = NeuralNetwork::Matrix(10, 1);
        for (int j = 0; j < x.training_images[i].size(); j++) {
            ip.value(j, 0) = (NeuralNetwork::real_nnt) x.training_images[i][j] / 255;
        }
        op.value(x.training_labels[i], 0) = 1;
        training_data.push_back(std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>(std::move(ip), std::move(op)));
    }

    for (int i = 0; i < 10000; i++) {
        NeuralNetwork::Matrix ip = NeuralNetwork::Matrix(784, 1);
        NeuralNetwork::Matrix op = NeuralNetwork::Matrix(10, 1);
        for (int j = 0; j < x.test_images[i].size(); j++) {
            ip.value(j, 0) = (NeuralNetwork::real_nnt)x.test_images[i][j] / 255;
        }
        op.value(x.test_labels[i], 0) = 1;
        testing_data.push_back(std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>(std::move(ip), std::move(op)));
    }
    
    NeuralNetwork::Network nn(std::vector<NeuralNetwork::size_nnt>({784, 64, 32, 10}), training_data, testing_data);
    

    auto device = cl_wrapper::get_devices(cl_wrapper::get_platforms_ids()[0])[0];
    auto context = cl_wrapper::create_context(device)
    nn.dump_to_file("../../../dumps/dump2.nn");
    auto dmp = read_file_bin("../../../dumps/dump2.nn");
    
    NeuralNetwork::Network recovered = NeuralNetwork::Network(dmp, training_data, testing_data);
    delete [] dmp;
    recovered.test();
    nn.test();
    
    return 0;
}