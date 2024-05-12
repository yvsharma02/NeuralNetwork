#pragma once

#include <vector>
#include <exception>
#include "matrix.h"
#include "types.h"


float ReLU(float f) {
    return f > 0 ? f : 0;
}

float ReLUDerivative(float f) {
    return f > 0 ? 1 : 0;
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

        real_nnt learning_rate = 0.01;
        std::vector<Matrix> weight_gradient_acculumator;
        std::vector<Matrix> bias_gradient_accumulator;

        public:
        Network(std::vector<size_nnt> layer_sizes, std::vector<std::pair<Matrix, Matrix>>&& training, std::vector<std::pair<Matrix, Matrix>>&& testing) : training(std::move(training)), testing(std::move(testing)) {
            for (int i = 0; i < layer_sizes.size(); i++) {
                activations.push_back(Matrix(layer_sizes[i], 1));
            }
            
            for (int i = 0; i < layer_sizes.size() - 1; i++) {
                z_activations.push_back(Matrix(layer_sizes[i + 1], 1));
                biases.push_back(Matrix(layer_sizes[i + 1], 1));
                errors.push_back(Matrix(layer_sizes[i + 1], 1));
                weights.push_back(Matrix(layer_sizes[i + 1], layer_sizes[i]));
                (--weights.end())->randomize();
                weight_gradient.push_back(Matrix(layer_sizes[i + 1], layer_sizes[i]));
            }
        }

        void train(int iterations, int batch_size) {
            while (iterations-- > 0) {

                std::vector<Matrix> weight_gradient_acculumator;
                std::vector<Matrix> bias_gradient_accumulator;

                for (int i = 0; i < weight_gradient.size(); i++) {
                    weight_gradient_acculumator[i] = Matrix(weight_gradient[i].row_count(), weight_gradient[i].col_count());
                }

                for (int i = 0; i < errors.size(); i++) {
                    bias_gradient_accumulator[i] = Matrix(errors[i].row_count(), errors[i].col_count());
                }
            
                for (int i = 0; i < training.size(); i++) {
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
            std::cout << "Correct: " << correct << "; Total" << testing.size() << "; Accuracy: " << (correct / testing.size()) << ";";
        }

        void foward_pass() {
            for (int i = 1; i < activations.size(); i++) {
                auto raw = weights[i - 1].multiply(activations[i - 1]);
                raw.add(biases[i - 1]);
                z_activations[i - 1] = std::move(raw);

                activations[i] = z_activations[i - 1].clone();
                activations[i].apply_function(&ReLU);
            }
        }

        void backward_pass() {
            for (int i = errors.size() - 2; i > 0; i--) {
                errors[i] = weights[i + 1].transpose().multiply(errors[i + 1]);
                errors[i].multiply_elementwise(z_activations[i]);
            }

            for (int i = weights.size() - 1; i >= 0; i--) {
                weight_gradient[i] = errors[i].multiply(activations[i]);
            }
        }

        void gradient_descent() {
            for (int i = 0; i < weights.size(); i++) {
                weights[i].add(weight_gradient_acculumator[i], -learning_rate, 1);
            }
            for (int i = 0; i < errors.size(); i++) {
                biases[i].add(bias_gradient_accumulator[i], -learning_rate, 1);
            }
        }


    };

};