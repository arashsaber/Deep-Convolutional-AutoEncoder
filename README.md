# Deep-Convolutional-AutoEncoder

This is a tutorial on creating a deep convolutional autoencoder with tensorflow.
The goal of the tutorial is to provide a simple template for convolutional autoencoders. Also, I value the use of tensorboard, and I hate it when the resulted graph and parameters of the model are not presented clearly in the tensorboard. Here, beside the main goal, I do my best to create a nice looking graph of the network on the tensorboard. The complete code can be found in the ConvolutionalAutoEncoder.py, further, brief explanation is presented in the wiki.

The layers are as follows:

coder part:
  - input layer
  - convolution
  - maxpool
  - drop out
  - fully connected
  - drop out
  - fully connected

Decoder part:
  - fully connected
  - drop out
  - fully connected
  - drop out
  - deconvolution
  - upsample
  - fully connected

We test the autoencoder on the MNIST database, and reduce the dimension of the inputs from 28*28 = 784 to 14*14 = 196 at the encoding layer.
