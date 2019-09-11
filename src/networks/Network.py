from abc import ABC , abstractmethod

import tensorflow
import tensorflow as tf



class Network(ABC):
    """
    Base class for network. Each networks needs a name to identify its weights and activating training
    when its weights get updated.
    """
    # placeholder to activate training
    training_placeholder = None
    # name to identify the network in the graph
    name = None

    def __init__(self,name,learning_rate):
        """

        :param name: string that identifies the network needs to be unique
        """
        self.training_placeholder  = tf.placeholder_with_default(False, shape=(), name=name + "is_training")
        self.name = name
        self.learning_rate = learning_rate

    def train_op(self, cost_function,
                 optimizer=None,
                 reuse=tf.AUTO_REUSE):
        """
        Creates an op inside the tensorflow graph to train the weights of the network

        :param cost_function: take the gradient of the cost function with respect to the weights of the network
        :param optimizer: what optimizer to use of the weight update
        :param reuse: if we want to reuse the parameters of the optimizer
        :return: an op to train the network
        """
        if optimizer is None:
            optimizer =tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            optimizer = optimizer
        trainable_variables = tf.trainable_variables()
        network_trainable_variables = [var for var in trainable_variables if var.name.startswith(self.name)]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops = [var for var in update_ops if var.name.startswith(self.name)]
        self._print_var_size(network_trainable_variables,self.name)
        with tf.variable_scope('train_vars', reuse=reuse):
            with tf.control_dependencies(update_ops):
                return optimizer.minimize(cost_function, var_list=network_trainable_variables)

    def get_training_placeholder(self):
        return self.training_placeholder

    @abstractmethod
    def tensor(self, input,xdim=-1, reuse=tf.AUTO_REUSE):
        """
        Returns a tensor that represents the output from the network given the input

        :param input: The input into the network
        :param xdim: The dimension of the output data. Networks with fixed output dimension will ignore this parameter
        :param reuse: If the input should be reused
        :return: a tensor
        """
        pass

    def _print_var_size(self,var,name):
        total_parameters = 0

        for variable in var:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            dim_list = []
            dim_list.append(variable.name)
            for dim in shape:
                variable_parameters *= dim.value
                dim_list.append(dim.value)
            total_parameters += variable_parameters
            print(dim_list)
        print("network: " +str(name) + " with size "+str(total_parameters))


