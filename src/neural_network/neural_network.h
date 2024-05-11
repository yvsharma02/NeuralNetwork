#pragma once

#include <vector>
#include <exception>
#include "matrix.h"
#include "types.h"

namespace NeuralNetwork {

    class LayerDescriptor {
        public:
        int raw_size;
        // Ignored for i/p layer.
        activation_func_nnt activation_func;
        activation_func_nnt derivative_func;

        int total_size(bool output_layer) {
            return raw_size + output_layer ? 0 : 1; // + 1 for bias.
        }
    };

    class Network {

        private:
        // activation function for 0th layer can be ignored.
        std::vector<LayerDescriptor> layer_descriptors;        
        std::vector<Matrix> layer_activations;
        // 0th will be empty. (input cannot have errors or z values.)
        std::vector<Matrix> z_activations;
        std::vector<Matrix> errors;

        std::vector<Matrix> weights;
        error_func_nnt error_func;

        public:
        Network(std::vector<LayerDescriptor> layers, error_func_nnt error_func) : error_func(error_func), layer_descriptors(layers) {
            if (layer_descriptors.size() < 2) {
                throw std::exception("NN must have atleast 2 layers");
            } 

            for (int i = 0; i < layer_descriptors.size(); i++) {
                layer_activations.push_back(Matrix(layer_descriptors[i].total_size(i == layer_descriptors.size() - 1), 1));
                // Set bias activation to always be 1.
                (--layer_activations.end())->value(layer_descriptors[i].raw_size, 1) = 1.0;

                errors.push_back(Matrix((--layer_activations.end())->row_count(), 1));
            }
            
            for (int i = 1; i < layer_descriptors.size(); i++) {
                weights.push_back(Matrix(layer_descriptors[i].total_size(i == layer_descriptors.size() - 1), layer_descriptors[i - 1].total_size(false)));
                (--weights.end())->randomize();
            }
        }

        void foward_pass() {
            for (int i = 1; i < layer_activations.size(); i++) {
                layer_activations[i] = weights[i - 1].multiply(layer_activations[i - 1]);
                for (int j = 0; j < layer_activations.size(); j++) {
                    layer_activations[i].value(j, 0) = layer_descriptors[i].activation_func(layer_activations[i].value(j, 0));
                }
            }
        }

        void backward_pass() {
            //TODO: Calculate error in last layer.

            for (int i = layer_activations.size() - 2; i >= 0; i--) {
                errors[i] = weights[x].transpose().multiply(errors[i + 1]).elementwise_multiply()
                // for (int j = 0; j < layer_activations[i].row_count(); i++) {
                //     errors[i].value(j, 0) = layer_activations[i].value(j, 0) - 
                // }
            }
        }

    };

};