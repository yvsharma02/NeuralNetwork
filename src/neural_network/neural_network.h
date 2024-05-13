#pragma once

#include <vector>
#include <exception>
#include "matrix.h"
#include "types.h"
#include "math.h"

#include "cl/cl.h"
#include "cl_wrapper/cl_helper.h"

#include <stdio.h>

float sigmoid(float f) {
    return 1.0 / (1 + expf(-f));
}

float sigmoid_derivative(float f) {
    return sigmoid(f) * (1 - sigmoid(f));
}

const char* read_kernel(const char* path = "../../../src/kernels/backprop.cl") {
    char buffer[10000];
    int c = 0;

    FILE* file = fopen(path, "r");
    char ch = fgetc(file);
    while (ch != EOF) {
        buffer[c++] = ch;
        ch = fgetc(file);
    }
    fclose(file);
    char* str = new char[c + 1];
    for (int i = 0; i < c; i++) {
        str[i] = buffer[i];
    }
    str[c] = 0;
//    std::cout << str << std::endl;
    return str;
}

namespace NeuralNetwork {

    class Network {

        private:
        // l0->w0->l1->w1->l2->w2->l3->w3->l4
        //         b0->    b1      b2      b3
        //         e0->    e1      e2      e3
        //         z0->    z1      z2      z3
        //     wg0     wg1


        std::vector<Matrix> z_activations;
        std::vector<Matrix> activations;
        std::vector<Matrix> errors;

        std::vector<Matrix> weight_gradient;

        std::vector<Matrix> weights;
        std::vector<Matrix> biases;

        std::vector<std::pair<Matrix, Matrix>> training;
        std::vector<std::pair<Matrix, Matrix>> testing;

        real_nnt learning_rate;
        std::vector<Matrix> weight_gradient_acculumator;
        std::vector<Matrix> bias_gradient_accumulator;

        public:
        Network(std::vector<size_nnt> layer_sizes, std::vector<std::pair<Matrix, Matrix>>&& training, std::vector<std::pair<Matrix, Matrix>>&& testing, real_nnt learning_rate = 0.05) : training(std::move(training)), testing(std::move(testing)), learning_rate(learning_rate) {
            for (int i = 0; i < layer_sizes.size(); i++) {
                activations.push_back(Matrix(layer_sizes[i], 1));
            }
            
            for (int i = 0; i < layer_sizes.size() - 1; i++) {
                z_activations.push_back(Matrix(layer_sizes[i + 1], 1));
                biases.push_back(Matrix(layer_sizes[i + 1], 1));
                errors.push_back(Matrix(layer_sizes[i + 1], 1));
                weights.push_back(Matrix(layer_sizes[i + 1], layer_sizes[i]));
                (--weights.end())->randomize(-0.75, 0.75);
                weight_gradient.push_back(Matrix(layer_sizes[i + 1], layer_sizes[i]));
            }

            for (int i = 0; i < weight_gradient.size(); i++) {
                weight_gradient_acculumator.push_back(Matrix(weight_gradient[i].row_count(), weight_gradient[i].col_count()));
            }

            for (int i = 0; i < errors.size(); i++) {
                bias_gradient_accumulator.push_back(Matrix(errors[i].row_count(), errors[i].col_count()));
            }
        }

        void train(int iterations, int batch_size) {
            while (iterations-- > 0) {

                for (int i = 0; i < weight_gradient.size(); i++) {
                    weight_gradient_acculumator[i].zeroify();
                }

                for (int i = 0; i < errors.size(); i++) {
                    bias_gradient_accumulator[i].zeroify();
                }
            
                for (int i = 0; i < training.size(); i++) {
                    std::cout << "Iteration: " << iterations << "; " << "Training: " << i << std::endl;
                    activations[0] = training[i].first.clone();
                    foward_pass();
                    errors[errors.size() - 1] = training[i].second.clone();
                    errors[errors.size() - 1].subtract(activations[activations.size() - 1]);
//                    if (training.size() - i <= 10) {
//                        training[i].second.print("Actual");
//                        activations[activations.size() - 1].print("Prediction");
//                    }
                    backward_pass();

                    for (int j = 0; j < weight_gradient.size(); j++) {
                        weight_gradient_acculumator[j].add(weight_gradient[j], 1.0 / batch_size, 1.0);
                    }

                    for (int j = 0; j < errors.size(); j++) {
                        bias_gradient_accumulator[j].add(errors[j], 1.0 / batch_size, 1.0);
                    }

                    if (i % batch_size == 0) {
                        gradient_descent();

                        for (int j = 0; j < weight_gradient.size(); j++) {
                            weight_gradient_acculumator[j].zeroify();
                        }

                        for (int j = 0; j < errors.size(); j++) {
                            bias_gradient_accumulator[j].zeroify();
                        }
                    }
                }
            }
        }

        void trainGPU(int iterations, int batch_size, cl_context& context, cl_device_id& device_id) {

            const char* strings_arr[] = { read_kernel() };
            size_t lens_arr[] = { strlen(strings_arr[0]) };

            auto program = clCreateProgramWithSource(context, 1, strings_arr, lens_arr, nullptr);
            cl_int res = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
            cl_int err;
            auto commandQueue = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
            
            while (iterations-- > 0) {
                std::cout << "Iteration: " << iterations << std::endl;
                auto kernal = clCreateKernel(program, "train", nullptr);
                int c = 0;
                
                int weights_size = 0;
                int biases_size = 0;
                int activations_size = 0;

                int *weights_sizes = new int[weights.size()];
                int *layer_sizes = new int[activations.size()];

                for (int i = 0; i < weights.size(); i++) {
                    weights_sizes[i] = weights[i].row_count() * weights[i].col_count();
                    weights_size += weights_sizes[i];
                }

                for (int i = 1; i < activations.size(); i++) {
                    biases_size += activations[i].row_count();
                }
                activations_size += biases_size + activations[0].row_count();

                for (int i = 0; i < activations.size(); i++) {
                    layer_sizes[i] = activations[i].row_count();
                }

                int input_size = training[0].first.row_count();
                int output_size = training[0].second.row_count();

                auto input_buffer_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real_nnt) * batch_size * input_size, nullptr, &err);
                auto output_buffer_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(real_nnt) * batch_size * output_size, nullptr, &err);
                auto weights_sizes_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * weights.size(), nullptr, &err);
                auto weights_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * weights_size, nullptr, &err);
                auto biases_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * biases_size, nullptr, &err);
                auto layer_sizes_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * activations.size() * batch_size, nullptr, &err);
                auto activations_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * activations_size * batch_size, nullptr, &err);
                auto z_activations_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * biases_size * batch_size, nullptr, &err);
                auto errors_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * biases_size * batch_size, nullptr, &err);
                auto wt_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * weights_size * batch_size, nullptr, &err);
                auto at_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * activations_size * batch_size, nullptr, &err);
                auto weight_gradient_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * weights_size * batch_size, nullptr, &err);

                
                for (int i = 0; i < batch_size; i++) {
                    auto ip_unrolled = training[c].first.unravel();
                    auto op_unrolled = training[c].second.unravel();
                    
                    err = clEnqueueWriteBuffer(commandQueue, input_buffer_d, CL_TRUE, input_size * sizeof(float) * i, input_size * sizeof(float), ip_unrolled, 0, nullptr, nullptr);
                    err = clEnqueueWriteBuffer(commandQueue, output_buffer_d, CL_TRUE, output_size * sizeof(float) * i, output_size * sizeof(float), op_unrolled, 0, nullptr, nullptr);
                    
                    if (err != 0) {
                        std::cout << ":(" << std::endl;
                    }
                    delete [] ip_unrolled;
                    delete [] op_unrolled;
                }

                int weight_sum = 0;
                for (int i = 0; i < weights.size(); i++) {
                    auto w = weights[i].unravel();
                    err = clEnqueueWriteBuffer(commandQueue, weights_d, CL_TRUE, weight_sum * sizeof(float), weights_sizes[i] * sizeof(float), w, 0, nullptr, nullptr);
                    weight_sum += weights_sizes[i];
                    delete[] w;
                }
//                weights[2].print("Normal");
                int bias_sum = 0;
                for (int i = 0; i < biases.size(); i++) {
                    auto w = biases[i].unravel();
                    err = clEnqueueWriteBuffer(commandQueue, biases_d, CL_TRUE, bias_sum * sizeof(float), biases[i].row_count() * sizeof(float), w, 0, nullptr, nullptr);
                    bias_sum += biases[i].row_count();
                    delete[] w;
                }

                err = clEnqueueWriteBuffer(commandQueue, weights_sizes_d, CL_TRUE, 0, sizeof(int) * weights.size(), weights_sizes, 0, nullptr, nullptr);
                err = clEnqueueWriteBuffer(commandQueue, layer_sizes_d, CL_TRUE, 0, sizeof(int) * activations.size(), layer_sizes, 0, nullptr, nullptr);

                err = clSetKernelArg(kernal, 0, sizeof(void*), &input_buffer_d);
                err = clSetKernelArg(kernal, 1, sizeof(void*), &output_buffer_d);
                err = clSetKernelArg(kernal, 2, sizeof(void*), &weights_sizes_d);
                err = clSetKernelArg(kernal, 3, sizeof(void*), &weights_d);
                err = clSetKernelArg(kernal, 4, sizeof(void*), &biases_d);
                int _size = activations.size();
                err = clSetKernelArg(kernal, 5, sizeof(int), &_size);
                err = clSetKernelArg(kernal, 6, sizeof(void*), &layer_sizes_d);
                err = clSetKernelArg(kernal, 7, sizeof(void*), &z_activations_d);
                err = clSetKernelArg(kernal, 8, sizeof(void*), &activations_d);
                err = clSetKernelArg(kernal, 9, sizeof(void*), &errors_d);
                err = clSetKernelArg(kernal, 10, sizeof(void*), &wt_d);
                err = clSetKernelArg(kernal, 11, sizeof(void*), &at_d);
                err = clSetKernelArg(kernal, 12, sizeof(void*), &weight_gradient_d);
                
                cl_event event;

                size_t global_groups_size = batch_size;
                const size_t local_groups_size = 32;
                //        err = clEnqueueTask (commandQueue, kernal, 0, nullptr, &event);
                err = clEnqueueNDRangeKernel(commandQueue, kernal, 1, nullptr, &global_groups_size, &local_groups_size, 0, nullptr, &event);
                clWaitForEvents(1, &event);

                real_nnt* weight_gradient_h = new float[weights_size * batch_size];
                real_nnt* bias_gradient_h = new float[biases_size * batch_size];
                real_nnt* wt_h = new float[weights_size * batch_size];
                
                err = clEnqueueReadBuffer(commandQueue, weight_gradient_d, CL_TRUE, 0, weights_size * sizeof(float), weight_gradient_h, 0, nullptr, nullptr);
                err = clEnqueueReadBuffer(commandQueue, biases_d, CL_TRUE, 0, biases_size  * sizeof(float), bias_gradient_h, 0, nullptr, nullptr);
                err = clEnqueueReadBuffer(commandQueue, wt_d, CL_TRUE, 0, weights_size * sizeof(float), wt_h, 0, nullptr, nullptr);

                int act_sum = 0;
                for (int i = 0; i < weights.size(); i++) {
                    if (i == 1) {
                        Matrix(wt_h, act_sum, weights[i].col_count(), weights[i].row_count()).print("WT");
                        weights[i].print("W");
                    }
                    act_sum += weights_sizes[i];
                }

                weight_sum = 0;
                for (int i = 0; i < weight_gradient_acculumator.size(); i++) {
                    weight_gradient_acculumator[i] = Matrix(weight_gradient_h, weight_sum, weights[i].row_count(), weights[i].col_count());
                    weight_sum += weights_sizes[i];
                }
//                weight_gradient_acculumator[2].print("WGA");


                bias_sum = 0;
                for (int i = 0; i < bias_gradient_accumulator.size(); i++) {
                    bias_gradient_accumulator[i] = Matrix(bias_gradient_h, bias_sum, biases[i].row_count(), biases[i].col_count());
                    bias_sum += biases[i].row_count();
//                    bias_gradient_accumulator[i].print("BGA");
                }

                err = clReleaseMemObject(input_buffer_d);
                err = clReleaseMemObject(output_buffer_d);
                err = clReleaseMemObject(weights_sizes_d);
                err = clReleaseMemObject(weights_d);
                err = clReleaseMemObject(biases_d);
                err = clReleaseMemObject(layer_sizes_d);
                err = clReleaseMemObject(z_activations_d);
                err = clReleaseMemObject(activations_d);
                err = clReleaseMemObject(errors_d);
                err = clReleaseMemObject(wt_d);
                err = clReleaseMemObject(at_d);
                err = clReleaseMemObject(weight_gradient_d);
                err = clReleaseKernel(kernal);

                delete[] weight_gradient_h;
                delete[] bias_gradient_h;
                delete[] weights_sizes;
                delete[] layer_sizes;
            }
        }

        void test() {
            int correct = 0;
            for (int i = 0; i < testing.size(); i++) {
                activations[0] = testing[i].first.clone();
                foward_pass();
                int temp;
                int actual;
                int predicted;
                activations[activations.size() - 1].max_index(predicted, temp);
                testing[i].second.max_index(actual, temp);
                correct += actual == predicted ? 1 : 0;
            }
            std::cout << "Correct: " << correct << "; Total: " << testing.size() << "; Accuracy: " << ((float)correct / testing.size()) * 100 << "%;";
        }

        void foward_pass() {
            for (int i = 1; i < activations.size(); i++) {
                Matrix raw = weights[i - 1].multiply(activations[i - 1]);
                raw.add(biases[i - 1]);
                z_activations[i - 1] = std::move(raw);

                activations[i] = z_activations[i - 1].clone();
                activations[i].apply_function(&sigmoid);
            }
        }

        void backward_pass() {
            for (int i = errors.size() - 2; i > 0; i--) {
                errors[i] = weights[i + 1].transpose().multiply(errors[i + 1]);
                auto derivative = z_activations[i].clone();
                derivative.apply_function(sigmoid_derivative);
                errors[i].multiply_elementwise(derivative);
            }
            for (int i = weights.size() - 1; i >= 0; i--) {
                weight_gradient[i] = errors[i].multiply(activations[i].transpose());
            }
        }

        void gradient_descent() {
            for (int i = 0; i < weights.size(); i++) {
                weights[i].add(weight_gradient_acculumator[i], learning_rate, 1);
            }
            for (int i = 0; i < errors.size(); i++) {
                biases[i].add(bias_gradient_accumulator[i], learning_rate, 1);
            }
        }


    };

};