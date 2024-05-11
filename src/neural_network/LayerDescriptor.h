#include <types.h>

namespace NeuralNetwork {

    class LayerDescriptor {
        public:
        int raw_size;
        // Ignored for i/p layer.
        activation_func_nnt activation_func;
        activation_func_nnt derivative_func;

        int total_size() {
            return raw_size + 1; // + 1 for bias.
        }
    };
}