#include <iostream>

#include "neural_network/matrix.h"
#include "neural_network/neural_network.h"
#include "reader/data_reader.h"
#include <cl/cl.h>
#include <stdlib.h>

 int index2D_to_1D(int i, int j, int cols) {
     return i * cols + j;
 }
// int index1D_to_2D(int x, int cols) {
//     return (x / cols) + (x % cols);
// }

/*

ABC
DEF
GHI
JKL
ABCDEFGHIJKL

ADGJ
BEHK
CFIL
ADGJBEHKCFIL

*/

// void transpose(float* a, int r, int c, float* res) {
//     for (int i = 0; i < r; i++) {
//         for (int j = 0; j < c; j++) {
//             res[j * r + i] = a[i * c + j];
//         }
//     }
// }

// void multiply_matrix(float* a, float* b, float* res, int res_r, int res_c, int common_len) {
//     for (int i = 0; i < res_r; i++) {
//         for (int j = 0; j < res_c; j++) {
//             float accumulator = 0.0;
//             for (int k = 0; k < common_len; k++) {
//                 accumulator += a[index2D_to_1D(i, k, common_len)] * b[index2D_to_1D(k, j, res_c)];
//             }
//             res[i * res_c + j] = accumulator;
//         }
//     }
// }

int main() {
//    srand(100);
    // int ll = 0;
    // NeuralNetwork::Matrix m(2, 3);
    // for (int i = 0; i < 2; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         m.value(i, j) = ll++;;
    //     }
    // }
    // auto n = m.transpose();
    // int km[6];
    // int kn[6];
    // for (int i = 0; i < 6; i++) {
    //     km[i] = m.unravel()[i];
    //     kn[i] = n.unravel()[i];
    // }

    // float res[4];
    // multiply_matrix(m.unravel(), n.unravel(), res, 2, 2, 3);

    // auto og = m.unravel();
    // float y[6];
    // for (int i = 0; i < 6; i++) {
    //     y[i] = og[i];
    // }
    // float res[6];
    // transpose(y, 2, 3, res);

    auto x = mnist::read_dataset();

    std::vector<std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>> training_data;
    std::vector<std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>> testing_data;

    for (int i = 0; i < 32 * 1867; i++) {
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
    NeuralNetwork::Network nn(std::vector<NeuralNetwork::size_nnt>({784, 20, 20, 10}), std::move(training_data), std::move(testing_data), 0.3);
    

    auto device = cl_wrapper::get_devices(cl_wrapper::get_platforms_ids()[0])[0];
    auto context = cl_wrapper::create_context(device);
    nn.trainGPU(15, 32, *context.context, device.device_id);
//    nn.train(10, 128);
    nn.test();
    
    return 0;
}