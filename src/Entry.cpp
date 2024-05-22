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

    for (int i = 0; i < 256 * 234; i++) {
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
    NeuralNetwork::Network nn(std::vector<NeuralNetwork::size_nnt>({784, 16, 16, 10}), std::move(training_data), std::move(testing_data));
    

    auto device = cl_wrapper::get_devices(cl_wrapper::get_platforms_ids()[0])[0];
    auto context = cl_wrapper::create_context(device);
//    nn.trainGPU(15, 256, *context.context, device.device_id, 1.55);
    nn.dump_to_file("../../../dumps/dump1.nn");
    nn.load_from_dump("../../../dumps/dump1.nn");
//    nn.train(10, 128, 1.35);
    nn.test();
    
    return 0;
}