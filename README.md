A simple feed-forward neural network, which can support multiple layers of varying sizes each.
Uses sigmoid activation for each layer. This can be easily changed, although chosing a different activaiton funcion for different layers may require some effort.

Backpropogation is implemented on GPU using OpenCL. A CPU based implementation is also present to make sure things are working correctly.
x items (specified by batch-size) are randomly sampled and loaded into the GPU memory.
Each of those x example undergoes forward pass and backward pass parallely on GPU.
The resulting weight matrices and biases matrices gradients are read from the GPU, and gradient desecent is performed on CPU. (Stochastic Gradient Descent)
Now another x samples are selected and the process repeats for epoch count.

Testing is currently completely done on CPU (Forward Pass is already implemented on GPU as part of backprogation, so implementing it on GPU should be straight-forward. )

Currently, MNIST database is used for training and testing
The code used to read the database is not my own, and is taken from:
https://github.com/wichtounet/mnist (MIT License)

Entire data is read at once (into main system memory), and may reqruire a large amount of memory. (Around ~4.6 GB)
Around 86% Accuracy was achieved while training using the following parameters:
15 iterations
64 batch size
1.25 learning rate.
2 hidden layers with 20 neurons in each.