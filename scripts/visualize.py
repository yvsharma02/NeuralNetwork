import struct
import numpy as np
from matplotlib import pyplot as plt

layers_count = 0
layer_sizes = []

bias_matrices = []
weight_matrices = []

with open("pre_trained_dump.bin", mode="rb") as file:
    content = file.read()
    pos = 0
    layers_count = int.from_bytes(content[0:8], byteorder='little', signed=False)
    layer_0_size = int.from_bytes(content[8:16], byteorder='little', signed=False)
    pos = 16

    layer_sizes.append(layer_0_size)
    while (pos < len(content)):
        for i in range(0, layers_count - 1):
            r = int.from_bytes(content[pos : pos + 8], byteorder='little', signed=False) 
            c = int.from_bytes(content[pos + 8: pos + 16], byteorder='little', signed=False) 
            pos += 16
            bias_matrices.append([struct.unpack('f', content[pos + 4 * j : pos + 4 * (j + 1)])[0] for j in range(0, r * c)])
            pos += r * c * 4
        for i in range(0, layers_count - 1):
            r = int.from_bytes(content[pos : pos + 8], byteorder='little', signed=False) 
            c = int.from_bytes(content[pos + 8: pos + 16], byteorder='little', signed=False) 
            pos += 16
            weight_matrices.append([struct.unpack('f', content[pos + 4 * j : pos + 4 * (j + 1)])[0] for j in range(0, r * c)])
            weight_matrices[len(weight_matrices) - 1] = [weight_matrices[len(weight_matrices) - 1][(k * c) : (k + 1) * c] for k in range(0, r)]
            pos += r * c * 4

fig, axs = plt.subplots(nrows = layers_count - 1, ncols = 2)

weight_matrices = [np.array(weight_matrices[i]) for i in range(0, len(weight_matrices))]
bias_matrices = [np.array(bias_matrices[i]).reshape(1, len(bias_matrices[i])) for i in range(0, len(bias_matrices))]

for i in range(0, layers_count - 1):

    axs[i, 0].imshow(weight_matrices[i], cmap='gray')
    axs[i, 0].set_title("Weight " + str(i))
    axs[i, 0].axis("off")

    axs[i, 1].imshow(bias_matrices[i], cmap='gray')
    axs[i, 1].set_title("Bias " + str(i))
    axs[i, 1].axis("off")

plt.show()