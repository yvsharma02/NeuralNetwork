A simple OpenCL based general purpose feed-forward multi-layerd neural network, which trains on GPU.
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
(Above example took around 200 sec to train) <br/>

While maintaining as similar conditions as possible, training on GPU resulted in approx 5x performance improvements.<br/>
(~10 mins on CPU vs ~2 min on GPU) (These times do not correspond to the settings mentioned above)<br/>
Training on i7 11800H (Single Thread, 16GB Memory)<br/>
RTX 3050 Laptop. (4GB)<br/>

The trained weights and biases can be dumped into a binary file, which can be loaded in consequent runs, to avoid trainng the model again. <br/>
The dump file can also be used in "visualize.py" scripts, which converts the weights and biases into grey-scale images. <br/> <br/>

The weights of biases of the trained network visualised as images: <br/>
(I hoped it will show some patterns or result in something more interesting than just random noise :( )

<img src="https://i.imgur.com/gacDtSb.png" width="512" height="256" />