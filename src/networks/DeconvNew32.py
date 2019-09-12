import tensorflow as tf

from networks.Network import Network


class DeconvNew32(Network):

    def __init__(self,name,learn_rate, layer_dim=1024,channels=1):
        super().__init__(name,learn_rate)
        self.channels = channels

    def tensor(self, input,xdim=-1, reuse=tf.AUTO_REUSE,channels=1):
        with tf.variable_scope(self.name, reuse=reuse):

            #4x4
            input = tf.layers.dense(input,units=4*4*256)
            input = tf.keras.layers.LayerNormalization()(input)
            input = tf.nn.relu(input)
            input = tf.reshape(input, (-1, 4, 4, 256))

            #8x8
            conv1 = tf.layers.conv2d_transpose(input,128,5,strides=[2,2],padding='same')
            conv1 = tf.keras.layers.LayerNormalization()(conv1)
            conv1 = tf.nn.relu(conv1)

            #16x16
            conv2 = tf.layers.conv2d_transpose(conv1,64,5,strides=[2,2],padding='same')
            conv2 = tf.keras.layers.LayerNormalization()(conv2)
            conv2 = tf.nn.relu(conv2)

            # 32x32
            output = tf.layers.conv2d_transpose(conv2, self.channels, 5,strides=[2,2], padding='same')

            return tf.nn.tanh(tf.reshape(output,(-1,32*32*self.channels)))