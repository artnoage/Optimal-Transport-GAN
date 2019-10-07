from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class Latent(ABC):
    """
    Abstract class for latent spaces.
    Subclasses only need to implement the sample() method.
    """

    @abstractmethod
    def __init__(self):
        """
        :param shape: The shape of a *single* sample.
        :param batch_size: The default batch size.
        """
        self.shape = None
        self.batch_size = None

    @abstractmethod
    def sample(self):
        """
        Returns several samples from the latent space.
        """
        pass


class Assignment_latent(Latent):

    def __init__(self, shape=None, batch_size=None):
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size
        self.initial_points = np.random.normal(0, 1, (50000, self.shape))
        self.fixed_latent = tf.constant(self.initial_points, dtype=tf.float32)
        self.name = "Assignment_latent"

    def sample(self, batch_size=None):
        raise NotImplemented

    def tensor(self):
        a = tf.random_shuffle(self.fixed_latent, seed=None, name=None)
        a = tf.slice(a, [0, 0], [self.batch_size, self.shape])
        return a + 0.02 * tf.random_normal(shape=(self.batch_size, self.shape))
