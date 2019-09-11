import tensorflow as tf
import numpy as np


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)


def clip_weights(network_name, bound=.01):
    """
    clips the weights to +/- the bound defined

    :param network_name:  name of the network weights that should be clipped
    :param bound: the bound to which the weights should be clipped
    :return: the clipped weights
    """
    trainable_variables = tf.trainable_variables()
    [print(var) for var in trainable_variables if var.op.name.startswith(network_name) and "batch" not in var.op.name ]
    weights = [var for var in trainable_variables if var.op.name.startswith(network_name)and "batch" not in var.op.name]
    #return [p.assign(tf.clip_by_value(p, -bound, bound)) for p in weights]

    with tf.name_scope('weight_clip'):
        clip_ops = []
        for weight in weights:
            clip_bounds = [-bound, bound]
            clip_ops.append(
                tf.assign(
                    weight,
                    tf.clip_by_value(weight, clip_bounds[0], clip_bounds[1])
                )
            )
        return tf.group(*clip_ops)


def convex_clip(network_name):
    """
    Clip the weights to e positive

    :param network_name: name of the network weights that should be clipped
    :return: the clipped weights
    """
    trainable_variables = tf.trainable_variables()
    weights = [var for var in trainable_variables if var.name.startswith(network_name)]
    with tf.name_scope('convex_clip'):
        clip_ops = []
        for weight in weights:
            clip_bounds = [0, np.inf]
            clip_ops.append(
                tf.assign(
                    weight,
                    tf.clip_by_value(weight, clip_bounds[0], clip_bounds[1])
                )
            )
        return tf.group(*clip_ops)

def norm_matrix(A,B):
    """
    Returns a matrix with the norms between each entry of A and B
    of shape BxA

    :param A: first matrix
    :param B: second matrix
    :return: norm matrix
    """
    epsilon = 1e-16
    return tf.norm(A - tf.expand_dims(B, axis=1)+epsilon,axis=2)


def y_tilde_vector(n,m):
    """
    Returns the y_tilde Matrix for the kernel distance approach returns a (n+m)x(n+m) matrix
    [0 .. m-1][0..m-1]  and entries are 1/m*1/m
    [m .. m+n-1][m .. m+n-1]  and entries are -1/n*-1/n
    rest is -1/n*1/m


    :param n: samples from the first distribution
    :param m: samples from the second distribution
    :return: the matrix
    """
    y_tilde_real = -(tf.ones(n) / tf.cast(n, dtype=tf.float32))
    y_tilde_fake = tf.ones(m) / tf.cast(m, dtype=tf.float32)
    y_tilde_combined = tf.concat([y_tilde_real, y_tilde_fake], axis=0)
    return tf.tensordot(y_tilde_combined, tf.transpose(y_tilde_combined), axes=0)



