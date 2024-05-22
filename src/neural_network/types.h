#include <vector>

namespace NeuralNetwork {
    typedef float real_nnt;
    typedef int size_nnt;
    typedef int int_nnt;

    typedef int32_t dump_type;

    typedef float (*activation_func_nnt)(float);

    typedef float (*error_func_nnt)(const std::vector<real_nnt>& expected, const std::vector<real_nnt>& predicted);
}