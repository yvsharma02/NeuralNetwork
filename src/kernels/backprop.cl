const int BATCH_SIZE = 50;
/* Structure of 1 element of io:
	Input Layer Activations [length = layer_sizes[0]]
	Expected Activcations [length = layer_sizes[layer_sizes.len - 1]]
*/

/* Structure of 1 element of weight: 
	W0 [layer_sizes[1]xlayer_sizes[0]]
	W1
	..
	..
*/

// int final_index(int layer, int index, int start, global const int* layer_sizes) {
// 	int c = 0;
// 	for (int i = 0; i < layer; i++) {
// 		c += layer_sizes[i];
// 	}

// 	return c + index + start;
// }

// int index2D_to_1D(int r, int c, int start, int row_size) {
// 	return start + (r * row_size) + c;
// }

// void multiply_matrix(global const float* a, global const float* b, global float* res,
// 					int res_r, int res_c, int common_len,
// 					int a_start, int b_start, int res_start,
// 					int a_row_size, int b_row_size, int res_row_size) {

// 	for (int i = 0; i < res_r; i++) {
// 		for (int j = 0; j < res_c; j++) {
// 			float accumulator = 0.0;
// 			for (int k = 0; k < common_len; k++) {
// 				accumulator += a[index2D_to_1D(i, k, a_start, a_row_size)] * b[index2D_to_1D(k, j, b_start, b_row_size)];
// 			}
// 			res[index2D_to_1D(i, j, res_start, res_row_size)] = accumulator;
// 		}
// 	}
// }

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

void multiply_elementwise(global const float* a, global const float* b, global float* res, int r, int c) {
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			res[index2D_to_1D(i, j, c)] += a[index2D_to_1D(i, j, c)] * b[index2D_to_1D(i, j, c)];
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

// void add_matrix(global float* add_into, global const float* b,
// 					int a_start, int b_start,
// 					int r, int c) {

// 	for (int i = 0; i < r; i++) {
// 		for (int j = 0; j < c; j++) {
// 			float accumulator = 0.0;
// 			add_into[index2D_to_1D(i, j, a_start, c)] += b[index2D_to_1D(i, j, b_start, c)];
// 		}
// 	}
// }

// void multiply_elementwise(global const float* a, global const float* b, global float* res,
// 					int res_r, int res_c, int common_len,
// 					int a_start, int b_start, int res_start,
// 					int a_row_size, int b_row_size, int res_row_size) {

// 	for (int i = 0; i < res_r; i++) {
// 		for (int j = 0; j < res_c; j++) {
// 			float accumulator = 0.0;
// 			res[index2D_to_1D(i, j, res_start, res_row_size)] = a[index2D_to_1D(i, j, a_start, a_row_size)] * b[index2D_to_1D(i, j, b_start, b_row_size)];
// 		}
// 	}
// }

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
				  global const float* weights, // get_weights_start
				  global const float* biases, // get_bias_start, does not store input llayer
				  const int layer_count,
				  global const int* layer_sizes,
// 				  global float* activation_buffer, // activation_start stores input layer
// //				  global float* wxa_buffer, // get_bias_start, does not store input layer
// 				  global float* wt_buffer, // get_weights_start
// 				  global float* raw_error_buffer, // wT x e, activations_start, does not store input layer
// 				  global float* at_buffer, // get_bias_start, does not store input layer
// 				  global float* wg_buffer // get_weights_start
// 				  global float* bias_gradient
				  global float* z_activations,
				  global float* activations,
				  global float* errors,
				  ) {
	// STEP 1: Move all buffers ahead to they start at the right position according to the current global_id
	int id = get_global_id(0);

	inputs += sizeof(float) * layer_sizes[0] * id;
	outputs += sizeof(float) * layer_sizes[layer_count - 1] * id;

	int weight_size = 0;
	int bias_size = 0;
	for (int i = 0; i < layer_count - 1; i++) {
		weight_size += weight_sizes[i];
		bias_size += layer_sizes[i + 1];
	}
	weights += sizeof(float) * weight_size * id;
	biases += sizeof(float) * bias_size * id;

//	wxa_buffer += sizeof(float) * (bias_size) * id;
	wt_buffer += sizeof(float) * weight_size * id;
	raw_error_buffer += sizeof(float) * (bias_size) * id;
	at_buffer += sizeof(float) * (bias_size + layer_sizes[0]) * id;
	wg_buffer += sizeof(float) * weight_size * id;
	activation_buffer += sizeof(float) * (bias_size + layer_sizes[0]) * id;
	bias_gradient += sizeof(float) * bias_size id;

	//STEP 2: Forward Pass
	for (int i = 0; i < layer_sizes[0]; i++) {
		activation_buffer[i] = inputs[i];
	}
	for (int i = 1; i < layer_count; i++) {
		multiply_matrix(get_weight_start(i - 1, weight_sizes, weights), get_activations_start(i - 1, layer_sizes, activation_buffer), 
		get_activations_start(i, layer_sizes, activations_buffer), layer_sizes[i], 1, layer_sizes[i - 1]);
		add_matrix(get_activations_start(i, layer_sizes, activations_buffer), get_bias_start(i - 1, layer_sizes, bias_start));
		apply_activation(get_activations_start(i, layer_sizes, activations_buffer), layer_sizes[i], 1);
	}

	//STEP 3: Backpropogation
	copy_matrix(get_bias_start(layer_count - 1), get_activations_start(layer_count - 1,));
	for (int i = 0; i < layer_sizes[0]; i++) {
	}
	for (int i = layer_sizes - 2; i > 0; i--) {
		
	}
}

// void kernel train(global const float* io, global const float* weights, global const float* biases,
// 					const int layer_count, global const int* layer_sizes, global const int* weight_matrix_sizes,
// 					global float* result_gradient, global float* result_bias_gradient, global float* activations_buffer) {

// 	int id = get_global_id(0);

// 	int input_start = (layer_sizes[0] + layer_sizes[layer_count - 1]) * id;
// 	int op_start = input_start + layer_sizes[0];

// 	int weights_size = 0;
// 	int bias_size = 0;
// 	int activations_size = bias_size;
// 	for (int i = 0; i < layer_count - 1; i++) {
// 		weights += weight_matrix_sizes[i]; //layer_sizes[i] * layer_sizes[i + 1];
// 		bias_size += layer_sizes[i + 1];
// 	}
// 	activations_size = bias_size + layer_sizes[0];

// 	int weights_start = weights_size * id;
// 	int bias_start = bias_size * id;

// 	int activations_buffer_start = (bias_size + layer_sizes[0]) * id;

// 	/* FORWARD PASS */
// 	for (int i = 0; i < layer_sizes[0]; i++) {
// 		activations_buffer[final_index(0, i, activations_buffer_start, layer_sizes)] = io[input_start + i];
// 	}

// 	int current_wb_start = 0;
// 	int current_activations_start = 0;
// 	for (int i = 1; i < layer_count; i++) {
// 		multiply_matrix(activations_buffer, weights, activations_buffer,
// 		layer_sizes[i], 1, layer_sizes[i - 1],
// 		activations_buffer_start, weights_start + current_wb_start, activations_buffer_start,
// 		1, layer_sizes[i - 1], 1
// 		);
// 		add_matrix(activations_buffer, biases, activations_buffer_start + current_activations_start, bias_start);
// 		current_wb_start += weight_matrix_sizes[i - 1];
// //		activations_buffer[final_index(1, i, activations_buffer_start, layer_sizes)] = io[input_start + i];
// 	}
// 	/* BACKWARD PASS */
// }