#pragma once

#include <vector>
#include "neural_network/matrix.h"

//typedef std::vector<std::pair<NeuralNetwork::Matrix, NeuralNetwork::Matrix>> training_set;
namespace NeuralNetwork {

    typedef float real_nnt;
    typedef size_t size_nnt;
    typedef int int_nnt;

    typedef int32_t dump_type;

    typedef float (*activation_func_nnt)(float);

    typedef float (*error_func_nnt)(const std::vector<real_nnt>& expected, const std::vector<real_nnt>& predicted);
}