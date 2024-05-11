#include <vector>

namespace NeuralNetwork {
    typedef double real_nnt;
    typedef size_t size_nnt;
    typedef int int_nnt;

    typedef float (*activation_func_nnt)(float);

    typedef float (*error_func_nnt)(const std::vector<real_nnt>& expected, const std::vector<real_nnt>& predicted);
}