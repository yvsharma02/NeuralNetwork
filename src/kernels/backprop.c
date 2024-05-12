#define global 
#define kernel
const int BATCH_SIZE = 50;

int index2D_to_1D(int i, int j, int cols) {
	return i * cols + j;
}

void multiply_matrix(global const float* a, global const float* b, global float* res, int res_r, int res_c, int common_len) {
	for (int i = 0; i < res_r; i++) {
		for (int j = 0; j < res_c; j++) {
			float accumulator = 0.0;
			for (int k = 0; k < common_len; k++) {
				accumulator += a[index2D_to_1D(i, k, res_r)] * b[index2D_to_1D(k, j, res_c)];
			}
			res[i * res_c + j] = accumulator;
		}
	}
}

void transpose(global float* a, int r, int c, global float* res) {
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			res[index2D_to_1D(j, i, r)] = res[index2D_to_1D(i, j, c)];
		}
	}
}

void multiply_elementwise(global float* a, global const float* b, /*global float* res,*/ int r, int c) {
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			a[index2D_to_1D(i, j, c)] *=/* a[index2D_to_1D(i, j, c)] * */b[index2D_to_1D(i, j, c)];
		}
	}
}

void add_matrix(global float* into, global float* b, int r, int c) {
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			into[index2D_to_1D(i, j, c)] += b[index2D_to_1D(i, j, c)];
		}
	}
}

void sub_matrix(global float* from, global float* to, int r, int c) {
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			from[index2D_to_1D(i, j, c)] -= to[index2D_to_1D(i, j, c)];
		}
	}
}

void copy_matrix(global float* to, global float* from, int  r, int c) {
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			to[index2D_to_1D(i, j, c)] = from[index2D_to_1D(i, j, c)];
		}
	}
}

const global float* get_weight_start(int layer, const global int* weight_sizes, const global float* start) {
	const global float* res = start;
	for (int i = 0; i < layer; i++) {
		res += weight_sizes[i];
	}

	return res;
}

const global float* get_activations_start(int layer, const global int* layer_sizes, const global float* start) {
	const global float* res = start;
	for (int i = 0; i < layer; i++) {
		res += layer_sizes[i];
	}

	return res;
}

float sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

const void apply_activation(global float* mat, int r, int c) {
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			mat[index2D_to_1D(i, j, c)] = sigmoid(index2D_to_1D(i, j, c));
		}
	}
}

const void apply_activation_derivative(global float* mat, int r, int c) {
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			mat[index2D_to_1D(i, j, c)] = sigmoid(index2D_to_1D(i, j, c)) * (1.0f - sigmoid(index2D_to_1D(i, j, c)));
		}
	}
}

// layer 0 -> first non input layer
const global float* get_bias_start(int layer, const global int* layer_sizes, const global float* start) {
	const global float* res = start;
	for (int i = 1; i < layer; i++) {
		res += layer_sizes[i];
	}

	return res;
}

void kernel train(global const float* inputs,
				  global const float* outputs,
				  global const int* weight_sizes,
				  global const float* weights,
				  global const float* biases,
				  const int layer_count,
				  global const int* layer_sizes,
				  global float* z_activations,
				  global float* activations,
				  global float* errors,
				  global float* wt,
				  global float* at,
				  global float* weight_gradient
				  ) {
	// STEP 1: Move all buffers ahead to they start at the right position according to the current global_id
	int id = get_global_id(0);

	inputs += sizeof(float) * layer_sizes[0] * id;
	outputs += sizeof(float) * layer_sizes[layer_count - 1] * id;

    // TODO: Bias, activation, errors may have different sizes.
	int weight_size = 0;
	int bias_size = 0;
	for (int i = 0; i < layer_count - 1; i++) {
		weight_size += weight_sizes[i];
		bias_size += layer_sizes[i + 1];
	}
    
	weights += sizeof(float) * weight_size * id;
	wt += sizeof(float) * weight_size * id;
	weight_gradient += sizeof(float) * weight_size * id;
	biases += sizeof(float) * bias_size * id;
	at += sizeof(float) * bias_size * id;

	z_activations += sizeof(float) * bias_size * id;
	activations += sizeof(float) * (bias_size + layer_sizes[0]) * id;
	errors += sizeof(float) * (bias_size) * id;

	//STEP 2: Forward Pass
	for (int i = 0; i < layer_sizes[0]; i++) {
		activations[i] = inputs[i];
	}
	for (int i = 1; i < layer_count; i++) {
		multiply_matrix(get_weight_start(i - 1, weight_sizes, weights), get_activations_start(i - 1, layer_sizes, activations), 
		get_activations_start(i - 1, layer_sizes, z_activations), layer_sizes[i], 1, layer_sizes[i - 1]);

		add_matrix(get_activations_start(i - 1, layer_sizes, z_activations), get_bias_start(i - 1, layer_sizes, biases), layer_sizes[i], 1);
		copy_matrix(get_activations_start(i, layer_sizes, activations), get_activations_start(i - 1, layer_sizes, z_activations), layer_sizes[i], 1);
		apply_activation(get_activations_start(i, layer_sizes, activations), layer_sizes[i], 1);
	}

	//STEP 3: Backpropogation
	//copy_matrix(get_bias_start(layer_count - 2, layer_sizes, errors), get_activations_start(layer_count - 1, layer_sizes, activations), layer_sizes[layer_count - 1], 1);
	//sub_matrix(get_bias_start(layer_count - 2, layer_sizes, errors), outputs, layer_sizes[layer_count - 1], 1);
    copy_matrix(get_bias_start(layer_count - 2, layer_sizes, errors), outputs, layer_sizes[layer_count - 1], 1);
    sub_matrix(get_bias_start(layer_count - 2, layer_sizes, errors), get_activations_start(layer_count - 1, layer_sizes, activations), layer_sizes[layer_count - 1], 1);
	
    for (int i = layer_count - 2; i > 0; i--) {
		transpose(get_weight_start(i + 1, weight_sizes, weights), layer_sizes[i + 1], layer_sizes[i], get_weight_start(i + 1, weight_sizes, wt));
		multiply_matrix(get_weight_start(i + 1, weight_sizes, wt), get_bias_start(i + 1, layer_sizes, errors), get_bias_start(i, layer_sizes, errors), layer_sizes[i], 1, layer_sizes[i + 1]);
		// i or i - 1?, layer_size?
        apply_activation_derivative(get_bias_start(i, layer_sizes, z_activations), layer_sizes[i], 1);
		multiply_elementwise(get_bias_start(i, layer_sizes, errors), get_bias_start(i, layer_sizes, z_activations), layer_sizes[i], 1);
	}

	for (int i = layer_count - 2; i >= 0; i--) {
		transpose(get_activations_start(i, layer_sizes, activations), layer_sizes[i], 1, get_activations_start(i, layer_sizes, at));
		multiply_matrix(get_bias_start(i, layer_sizes, errors), get_activations_start(i, layer_sizes, at),
        get_weight_start(i, weight_sizes, weight_gradient), layer_sizes[i + 1], layer_sizes[i], 1);
	}
}
