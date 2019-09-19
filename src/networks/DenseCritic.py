import numpy as np
import tensorflow as tf

from networks.Network import Network


class DenseCritic(Network):

    def __init__(self, name=None, learn_rate=None, layer_dim=None,xdim=20):
        super().__init__(name, learn_rate)
        self.layer_dim = layer_dim
        self.xdim=xdim
    def tensor(self, input,xdim=-1, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=reuse):
            inputs = tf.reshape(input, shape=(-1, self.xdim))
            output = tf.layers.dense(inputs, self.layer_dim, activation=tf.nn.leaky_relu)
            output = tf.layers.dense(output, self.layer_dim, activation=tf.nn.leaky_relu)
            output = tf.layers.dense(output, self.layer_dim, activation=tf.nn.leaky_relu)
            output = tf.layers.dense(output, self.layer_dim, activation=tf.nn.leaky_relu)
            output = tf.layers.dense(output, 1,)
            output = tf.reshape(output, [-1, 1])
        return output
