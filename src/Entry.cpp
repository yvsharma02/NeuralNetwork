#include <iostream>

#include "neural_network/matrix.h"
#include "neural_network/neural_network.h"
#include "reader/data_reader.h"
#include <cl/cl.h>



int main() {
    auto x = mnist::read_dataset();

    std::vector<std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>> training_data;
    std::vector<std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>> testing_data;

    for (int i = 0; i < 1024 * 10; i++) {
        NeuralNetwork::Matrix ip = NeuralNetwork::Matrix(784, 1);
        NeuralNetwork::Matrix op = NeuralNetwork::Matrix(10, 1);
        for (int j = 0; j < x.training_images[i].size(); j++) {
            ip.value(j, 0) = (NeuralNetwork::real_nnt) x.training_images[i][j] / 255;
        }
        op.value(x.training_labels[i], 0) = 1;
       // op.print();
        training_data.push_back(std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>(std::move(ip), std::move(op)));
    }

    for (int i = 0; i < 10000; i++) {
        NeuralNetwork::Matrix ip = NeuralNetwork::Matrix(784, 1);
        NeuralNetwork::Matrix op = NeuralNetwork::Matrix(10, 1);
        for (int j = 0; j < x.test_images[i].size(); j++) {
            ip.value(j, 0) = (NeuralNetwork::real_nnt)x.test_images[i][j] / 255;
        }
        op.value(x.test_labels[i], 0) = 1;
//        op.print();
        testing_data.push_back(std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>(std::move(ip), std::move(op)));
    }
    NeuralNetwork::Network nn(std::vector<NeuralNetwork::size_nnt>({784, 20, 20, 10}), std::move(training_data), std::move(testing_data), 0.1);
    

    auto device = cl_wrapper::get_devices(cl_wrapper::get_platforms_ids()[0])[0];
    auto context = cl_wrapper::create_context(device);
    nn.trainGPU(10, 128, *context.context, device.device_id);
//    nn.train(10, 25);
    nn.test();
    
    return 0;
}