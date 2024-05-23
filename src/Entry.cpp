#include <iostream>

#include "neural_network/matrix.h"
#include "neural_network/neural_network.h"
#include "reader/data_reader.h"
#include <cl/cl.h>
#include <stdlib.h>

int main() {
    srand(1234);

    const int batch_size = 64;
    const int testing_size = 60000 / batch_size;

    std::vector<std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>> training_data;
    std::vector<std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>> testing_data;
    
    {
        std::cout << "Reading Data" << std::endl;
        auto dataset = mnist::read_dataset();

        std::cout << "Constructing Data" << std::endl;
        for (int i = 0; i < batch_size * testing_size; i++) {
            NeuralNetwork::Matrix ip = NeuralNetwork::Matrix(784, 1);
            NeuralNetwork::Matrix op = NeuralNetwork::Matrix(10, 1);
            for (int j = 0; j < dataset.training_images[i].size(); j++) {
                ip.value(j, 0) = (NeuralNetwork::real_nnt) dataset.training_images[i][j] / 255;
            }
            op.value(dataset.training_labels[i], 0) = 1;
            training_data.push_back(std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>(std::move(ip), std::move(op)));
        }

        for (int i = 0; i < 10000; i++) {
            NeuralNetwork::Matrix ip = NeuralNetwork::Matrix(784, 1);
            NeuralNetwork::Matrix op = NeuralNetwork::Matrix(10, 1);
            for (int j = 0; j < dataset.test_images[i].size(); j++) {
                ip.value(j, 0) = (NeuralNetwork::real_nnt)dataset.test_images[i][j] / 255;
            }
            op.value(dataset.test_labels[i], 0) = 1;
            testing_data.push_back(std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>(std::move(ip), std::move(op)));
        }
    }

    auto device = cl_wrapper::get_devices(cl_wrapper::get_platforms_ids()[0])[0];
    auto context = cl_wrapper::create_context(device);
    
    const char* dump_path = "../../../pre_trained_dump.bin";
    /*
    NeuralNetwork::Network original(std::vector<NeuralNetwork::size_nnt>({784, 64, 32, 10}));
    std::cout << "Training" << std::endl;
    original.trainGPU(7, batch_size, *context.context, device.device_id, 5.2, training_data);

    original.dump_to_file(dump_path);
    */

    NeuralNetwork::Network recovered = NeuralNetwork::Network::create_form_dump(dump_path);

    std::cout << "Testing" << std::endl;
//    original.test(testing_data);
    recovered.test(testing_data);
    std::cout << "Cleaning Up." << std::endl;

    return 0;
}