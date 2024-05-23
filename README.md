A simple feed-forward neural network, which can support multiple layers of varying sizes each.
Uses sigmoid activation for each layer. This can be easily changed, although chosing a different activaiton funcion for different layers may require some effort.

Backpropogation is implemented on GPU using OpenCL. A CPU based implementation is also present to make sure things are working correctly. <br/>
x items (specified by batch-size) are randomly sampled and loaded into the GPU memory.<br/>
Each of those x example undergoes forward pass and backward pass parallely on GPU.<br/>
The resulting weight matrices and biases matrices gradients are read from the GPU, and gradient desecent is performed on CPU. (Stochastic Gradient Descent).<br/>
Now another x samples are selected and the process repeats for epoch count.<br/>

Testing is currently completely done on CPU (Forward Pass is already implemented on GPU as part of backprogation, so implementing it on GPU should be straight-forward. )<br/>

Currently, MNIST database is used for training and testing<br/>
The code used to read the database is not my own, and is taken from:<br/>
https://github.com/wichtounet/mnist (MIT License)<br/>

Entire data is read at once (into main system memory), and may reqruire a large amount of memory. (Around ~4.6 GB)<br/>
Around 90% Accuracy was achieved while training using the following parameters:<br/>
15 iterations<br/>
64 batch size<br/>
1.35 learning rate.<br/>
2 hidden layers with 64 and 32 neurons respectively.<br/>

Adding a 3rd layer started to result in overfitting.<br/>

While maintaining as similar conditions as possible, training on GPU resulted in approx 5x performance improvements.<br/>
(~10 mins on CPU vs ~2 min on GPU) (These times do not correspond to the settings mentioned above)<br/>
Training on i7 11800H (Single Thread)<br/>
RTX 3050 Laptop.<br/>
