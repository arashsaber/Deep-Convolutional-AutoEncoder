"""
Deep Convolutional Autoencoder with TensorFlow

Arash Saber Tehrani - May 2017

"""
#   ---------------------------------
# import required packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#   ---------------------------------
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 100

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# This is
logs_path = "./logs/"
#   ---------------------------------
"""
We start by creating the layers with name scopes so that the graph in
the tensorboard looks meaningful
"""
#   ---------------------------------
def conv2d(input, name, kshape, strides=[1, 1, 1, 1]):
    with tf.name_scope(name):
        W = tf.get_variable(name='w_'+name,
                            shape=kshape,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        out = tf.nn.conv2d(input,W,strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, b)
        out = tf.nn.relu(out)
        return out
# ---------------------------------
def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
    with tf.name_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                                                 num_outputs= n_outputs,
                                                 kernel_size=kshape,
                                                 stride=strides,
                                                 padding='SAME',
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                                 biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                 activation_fn=tf.nn.relu)
        return out
#   ---------------------------------
def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape, #size of window
                             strides=strides,
                             padding='SAME')
        return out
#   ---------------------------------
def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out
#   ---------------------------------
def fullyConnected(input, name, output_size):
    with tf.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='w_'+name,
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_'+name,
                            shape=[output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        input = tf.reshape(input, [-1, input_size])
        out = tf.nn.relu(tf.add(tf.matmul(input, W), b))
        return out
#   ---------------------------------
def dropout(input, name, keep_rate):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, keep_rate)
        return out
#   ---------------------------------
# Let us now design the autoencoder
def ConvAutoEncoder(x, name):
    with tf.name_scope(name):
        """
        We want to get dimensionality reduction of 784 to 196
        Layers:
            input --> 28, 28 (784)
            conv1 --> kernel size: (5,5), n_filters:25 ???make it small so that it runs fast
            pool1 --> 14, 14, 25
            dropout1 --> keeprate 0.8
            reshape --> 14*14*25
            FC1 --> 14*14*25, 14*14*5
            dropout2 --> keeprate 0.8
            FC2 --> 14*14*5, 196 --> output is the encoder vars
            FC3 --> 196, 14*14*5
            dropout3 --> keeprate 0.8
            FC4 --> 14*14*5,14*14*25
            dropout4 --> keeprate 0.8
            reshape --> 14, 14, 25
            deconv1 --> kernel size:(5,5,25), n_filters: 25
            upsample1 --> 28, 28, 25
            FullyConnected (outputlayer) -->  28* 28* 25, 28 * 28
            reshape --> 28*28
        """
        input = tf.reshape(x, shape=[-1, 28, 28, 1])

        # coding part
        c1 = conv2d(input, name='c1', kshape=[5, 5, 1, 25])
        p1 = maxpool2d(c1, name='p1')
        do1 = dropout(p1, name='do1', keep_rate=0.75)
        do1 = tf.reshape(do1, shape=[-1, 14*14*25])
        fc1 = fullyConnected(do1, name='fc1', output_size=14*14*5)
        do2 = dropout(fc1, name='do2', keep_rate=0.75)
        fc2 = fullyConnected(do2, name='fc2', output_size=14*14)
        # Decoding part
        fc3 = fullyConnected(fc2, name='fc3', output_size=14 * 14 * 5)
        do3 = dropout(fc3, name='do3', keep_rate=0.75)
        fc4 = fullyConnected(do3, name='fc4', output_size=14 * 14 * 25)
        do4 = dropout(fc4, name='do3', keep_rate=0.75)
        do4 = tf.reshape(do4, shape=[-1, 14, 14, 25])
        dc1 = deconv2d(do4, name='dc1', kshape=[5,5],n_outputs=25)
        up1 = upsample(dc1, name='up1', factor=[2, 2])
        output = fullyConnected(up1, name='output', output_size=28*28)
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.square(tf.subtract(output, x)))
        return output, cost
#   ---------------------------------
def train_network(x):
    prediction, cost = ConvAutoEncoder(x, 'ConvAutoEnc')
    with tf.name_scope('opt'):
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Create a summary to monitor cost tensor
    tf.summary.scalar("cost", cost)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    n_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # create log writer object
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        for epoch in range(n_epochs):
            avg_cost = 0
            n_batches = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(n_batches):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_x, y: batch_y})
                # Compute average loss
                avg_cost += c / n_batches
                # write log
                writer.add_summary(summary, epoch * n_batches + i)

            # Display logs per epoch step
            print('Epoch', epoch+1, ' / ', n_epochs, 'cost:', avg_cost)
        print('Optimization Finished')
        print('Cost:', cost.eval({x: mnist.test.images}))


train_network(x)




