from networks.NetworkABC import NetworkABC

import tensorflow as tf


class Conv32(NetworkABC):


    def __init__(self,name,learn_rate, layer_dim=1024,channels=1):
        super().__init__(name,learn_rate)
        self.channels = channels

    def tensor(self, input,xdim=-1, reuse=tf.AUTO_REUSE,channels=1):

        with tf.variable_scope(self.name, reuse=reuse):
            # make sure its the correct format
            inputs = tf.reshape(input, (-1, 32, 32, self.channels))


            # 16 x16
            conv0 = tf.layers.conv2d(inputs, 64, 5, strides=(2, 2), padding='same')
            bn_0 = tf.keras.layers.LayerNormalization()(conv0)
            lrelu_0 = tf.nn.leaky_relu(bn_0)

            #8x8
            conv1 = tf.layers.conv2d(lrelu_0, 128, 5, strides=(2, 2), padding='same')
            bn_1 = tf.keras.layers.LayerNormalization()(conv1)
            lrelu_1 = tf.nn.leaky_relu(bn_1)

            #4x4
            conv_2 = tf.layers.conv2d(lrelu_1, 256, 5, strides=(2, 2), padding='same')
            bn_2 =tf.keras.layers.LayerNormalization()(conv_2)
            lrelu_2 = tf.nn.leaky_relu(bn_2)

            flatten_2 = tf.layers.flatten(lrelu_2)
            output = tf.layers.dense(flatten_2,1)
            return output
