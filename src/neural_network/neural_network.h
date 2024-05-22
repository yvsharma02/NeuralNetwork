#pragma once

#include <vector>
#include <exception>
#include "matrix.h"
#include "types.h"
#include "math.h"
#include <stdlib.h>
#include "cl/cl.h"
#include "cl_wrapper/cl_helper.h"

#include <stdio.h>

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

        std::vector<Matrix> weight_gradient_acculumator;
        std::vector<Matrix> bias_gradient_accumulator;

        public:
        Network(std::vector<size_nnt> layer_sizes, std::vector<std::pair<Matrix, Matrix>>&& training, std::vector<std::pair<Matrix, Matrix>>&& testing) : training(std::move(training)), testing(std::move(testing)) {
            init(layer_sizes);
        }

        void init(const std::vector<size_nnt>& layer_sizes, bool clean_build = true) {
            for (int i = 0; i < layer_sizes.size(); i++) {
                activations.push_back(Matrix(layer_sizes[i], 1));
            }

            for (int i = 0; i < layer_sizes.size() - 1; i++) {
                z_activations.push_back(Matrix(layer_sizes[i + 1], 1));
                errors.push_back(Matrix(layer_sizes[i + 1], 1));
                if (clean_build) {
                    biases.push_back(Matrix(layer_sizes[i + 1], 1));
                    weights.push_back(Matrix(layer_sizes[i + 1], layer_sizes[i]));
                    (--weights.end())->randomize(-0.75, 0.75);
                }
                weight_gradient.push_back(Matrix(layer_sizes[i + 1], layer_sizes[i]));
            }

            for (int i = 0; i < weight_gradient.size(); i++) {
                weight_gradient_acculumator.push_back(Matrix(weight_gradient[i].row_count(), weight_gradient[i].col_count()));
            }

            for (int i = 0; i < errors.size(); i++) {
                bias_gradient_accumulator.push_back(Matrix(errors[i].row_count(), errors[i].col_count()));
            }
        }

        void dump_to_file(const char* path) const {
            size_t size;
            auto dmp = dump(size);
            write_file(dmp, 0, size, path);
            delete [] dmp;
        }

        void load_from_file(const char* path) {
            const char* arr = read_file(path);
            load_from_dump(arr);
            delete [] arr;
        }

        char* dump(size_t& total_size) const {
            // Layer Count
            // 1st layer size.
            // Bias Matrices.
            // Weight Matrices

            total_size = sizeof(size_nnt); // for layer count.
            total_size += sizeof(size_nnt); // for the size of 1st (input) layer.
            for (int i = 0; i < weights.size(); i++) {
                total_size += weights[i].dump_size();
            }
            for (int i = 0; i < biases.size(); i++) {
                total_size += biases[i].dump_size();
            }
            
        
            char* dump = new char[total_size];
            int c = 0;
            
            size_nnt activations_size = activations.size();
            memcpy(dump + c, &activations_size, sizeof(size_nnt));
            c += sizeof(size_nnt);
            
            size_nnt layer_0_size = activations[0].row_count();
            memcpy(dump + c, &layer_0_size, sizeof(size_nnt));
            c += sizeof(size_nnt);
            
            for (int i = 0; i < biases.size(); i++) {
                biases[i].dump(dump, c);
                c += biases[i].dump_size();
            }
            for (int i = 0; i < weights.size(); i++) {
                weights[i].dump(dump, c);
                c += weights[i].dump_size();
            }

            return dump;
        }

        void load_from_dump(const char* dump) {

            this->~Network();
            
            int c = 0;
            size_nnt layer_counts = 0;// ((size_nnt*)dump)[0];
            memcpy(&layer_counts , dump + c, sizeof(size_nnt));
            c += sizeof(size_nnt);

//            layer_counts = std::vector<size_nnt>();

            std::vector<size_nnt> layer_sizes;
            
            size_nnt layer_0_size;
            memcpy(&layer_0_size, dump + c, sizeof(size_nnt));
            c += sizeof(size_nnt);
            layer_sizes.push_back(layer_0_size);

            for (int i = 0; i < layer_counts - 1; i++) {
                biases.push_back(Matrix(0, 0));
                c += biases[i].load_from_dump(dump, c);
                layer_sizes.push_back(biases[i].row_count());
            }

            for (int i = 0; i < layer_counts - 1; i++) {
                weights.push_back(Matrix(0,0));
                c += weights[i].load_from_dump(dump, c);
            }
            init(layer_sizes, false);
            
        }

        void train(int iterations, int batch_size, real_nnt learning_rate) {
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
                    backward_pass();

                    for (int j = 0; j < weight_gradient.size(); j++) {
                        weight_gradient_acculumator[j].add(weight_gradient[j], 1.0 / batch_size, 1.0);
                    }

                    for (int j = 0; j < errors.size(); j++) {
                        bias_gradient_accumulator[j].add(errors[j], 1.0 / batch_size, 1.0);
                    }

                    if (i % batch_size == 0) {
                        gradient_descent(learning_rate);

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

        void trainGPU(int epoch_count, int batch_size, cl_context& context, cl_device_id& device_id, real_nnt learning_rate, bool random_sampling = true) {
            if (training.size() % batch_size != 0) {
                throw std::exception("Training Size must be a multiple of batch size");
            }
            const char* strings_arr[] = { read_file("../../../src/kernels/backprop.cl") };
            size_t lens_arr[] = { strlen(strings_arr[0]) };

            auto program = clCreateProgramWithSource(context, 1, strings_arr, lens_arr, nullptr);
            cl_int res = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
            cl_int err;

            delete[] strings_arr[0];

            auto commandQueue = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
            
            int weights_size = 0;
            int biases_size = 0;
            int activations_size = 0;

            int* weights_sizes = new int[weights.size()];
            int* layer_sizes = new int[activations.size()];

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
            auto at_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * (activations_size - layer_sizes[activations.size() - 1]) * batch_size, nullptr, &err);
            auto weight_gradient_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * weights_size * batch_size, nullptr, &err);

            err = clEnqueueWriteBuffer(commandQueue, weights_sizes_d, CL_TRUE, 0, sizeof(int) * weights.size(), weights_sizes, 0, nullptr, nullptr);
            err = clEnqueueWriteBuffer(commandQueue, layer_sizes_d, CL_TRUE, 0, sizeof(int) * activations.size(), layer_sizes, 0, nullptr, nullptr);

            while (epoch_count-- > 0) {
                for (int l = 0; l < training.size() / batch_size; l++) {
                    std::cout << "Iteration: " << epoch_count << "; Batch: " << l << std::endl;
                    for (int i = 0; i < weight_gradient_acculumator.size(); i++) {
                        weight_gradient_acculumator[i].zeroify();
                    }
                    for (int i = 0; i < bias_gradient_accumulator.size(); i++) {
                        bias_gradient_accumulator[i].zeroify();
                    }

                    auto kernal = clCreateKernel(program, "train", nullptr);
                    int sample_c = 0;
                    for (int i = 0; i < batch_size; i++) {
                        sample_c = sample_c % training.size();
                        int rand = random_sampling ? rand_int(training.size()) : sample_c++;
                        auto ip_unrolled = training[rand].first.unravel();
                        auto op_unrolled = training[rand].second.unravel();

                        err = clEnqueueWriteBuffer(commandQueue, input_buffer_d, CL_TRUE, input_size * sizeof(float) * i, input_size * sizeof(float), ip_unrolled, 0, nullptr, nullptr);
                        err = clEnqueueWriteBuffer(commandQueue, output_buffer_d, CL_TRUE, output_size * sizeof(float) * i, output_size * sizeof(float), op_unrolled, 0, nullptr, nullptr);

                        delete[] ip_unrolled;
                        delete[] op_unrolled;
                    }

                    int weight_sum = 0;
                    for (int i = 0; i < weights.size(); i++) {
                        auto w = weights[i].unravel();
                        err = clEnqueueWriteBuffer(commandQueue, weights_d, CL_TRUE, weight_sum * sizeof(float), weights_sizes[i] * sizeof(float), w, 0, nullptr, nullptr);
                        weight_sum += weights_sizes[i];
                        delete[] w;
                    }
                    int bias_sum = 0;
                    for (int i = 0; i < biases.size(); i++) {
                        auto w = biases[i].unravel();
                        err = clEnqueueWriteBuffer(commandQueue, biases_d, CL_TRUE, bias_sum * sizeof(float), biases[i].row_count() * sizeof(float), w, 0, nullptr, nullptr);
                        bias_sum += biases[i].row_count();
                        delete[] w;
                    }

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
                    err = clEnqueueNDRangeKernel(commandQueue, kernal, 1, nullptr, &global_groups_size, &local_groups_size, 0, nullptr, &event);
                    clWaitForEvents(1, &event);

                    real_nnt* weight_gradient_h = new float[weights_size * batch_size];
                    real_nnt* bias_gradient_h = new float[biases_size * batch_size];


                    err = clEnqueueReadBuffer(commandQueue, weight_gradient_d, CL_TRUE, 0, weights_size * sizeof(float), weight_gradient_h, 0, nullptr, nullptr);
                    err = clEnqueueReadBuffer(commandQueue, errors_d, CL_TRUE, 0, biases_size * sizeof(float), bias_gradient_h, 0, nullptr, nullptr);

                    weight_sum = 0;
                    for (int i = 0; i < weight_gradient_acculumator.size(); i++) {
                        auto x = Matrix(weight_gradient_h, weight_sum, weights[i].row_count(), weights[i].col_count());

                        weight_gradient_acculumator[i].add(x, 1.0 / batch_size, 1);
                        weight_sum += weights_sizes[i];
                    }

                    bias_sum = 0;
                    for (int i = 0; i < bias_gradient_accumulator.size(); i++) {
                        auto x = Matrix(bias_gradient_h, bias_sum, biases[i].row_count(), biases[i].col_count());
                        bias_gradient_accumulator[i].add(x , 1.0 / batch_size, 1);
                        bias_sum += biases[i].row_count();
                    }

                    err = clReleaseKernel(kernal);

                    gradient_descent(learning_rate);
                    delete[] weight_gradient_h;
                    delete[] bias_gradient_h;
                }
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
            

            delete[] weights_sizes;
            delete[] layer_sizes;
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

        void gradient_descent(real_nnt learning_rate) {
            for (int i = 0; i < weights.size(); i++) {
                weights[i].add(weight_gradient_acculumator[i], learning_rate, 1);
            }
            for (int i = 0; i < errors.size(); i++) {
                biases[i].add(bias_gradient_accumulator[i], learning_rate, 1);
            }
        }


    };

};