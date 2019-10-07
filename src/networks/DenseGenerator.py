import numpy as np
import tensorflow as tf

from networks.NetworkABC import NetworkABC


class DenseGenerator(NetworkABC):

    def __init__(self, name=None, learn_rate=None, layer_dim=1024, xdim=20):
        super().__init__(name, learn_rate)
        self.layer_dim = layer_dim
        self.xdim=xdim


    def tensor(self, input, xdim=-1, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name,reuse=reuse):
            output = tf.layers.dense(input, self.layer_dim, activation=tf.nn.leaky_relu,)
            #output = tf.layers.batch_normalization(output)
            output=tf.keras.layers.LayerNormalization()(output)
            output = tf.layers.dense(output, self.layer_dim, activation=tf.nn.leaky_relu)
            #output = tf.layers.batch_normalization(output)
            output=tf.keras.layers.LayerNormalization()(output)
            output = tf.layers.dense(output, self.layer_dim, activation=tf.nn.leaky_relu)
            #output = tf.layers.batch_normalization(output)
            output=tf.keras.layers.LayerNormalization()(output)
            output = tf.layers.dense(output, self.xdim, activation=tf.nn.tanh)
        return output

