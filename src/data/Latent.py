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

# This is the standard way to generate points. All points are generated by one Gaussian.

class Gaussian_latent(Latent):

    def __init__(self, shape=None, batch_size=None):
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size
        self.name = "Tensorflow_latent"
    def sample(self, batch_size=None):
        raise NotImplemented

    def tensor(self):
        return tf.random_normal(shape=(self.batch_size, self.shape))

# With this method, we generate random points in our latent space. The points are taken from N Gaussians with some Variance.
# If one makes N too big then we have a perfect fit, or an overfit depending how one sees it.
# If the number of Gaussians is small then it is harder to train and someone has to increase the number of critic steps and variance to avoid mode collapse.
# However we expect that if the dataset allows it, then small N of Gaussians will result in some clustering of the data.
# We also prefer this method from the standard one, because we believe that quite often the data manifold has different geometry in different parts.
# In the case of Fashion Mnist one needs more Gaussians than the dataset itself. We believe that this happens because the Mnist databases dont really have
# any accumulation points, at least with the Euclidean distance.
# If for your data set, you observe that you have many non_assigned points, either increase the factor in front of the noise or the number of Gaussians.

class Assignment_latent(Latent):

    def __init__(self, shape=None, batch_size=None):
        super().__init__()
        N_Gaussians=10000
        self.shape = shape
        self.batch_size = batch_size
        self.initial_points=np.random.normal(0,1,(N_Gaussians,self.shape))
        self.fixed_latent = tf.constant(self.initial_points, dtype=tf.float32)
        self.name = "Assignment_latent"
        self.variance=0.1
    def sample(self, batch_size=None):
        raise NotImplemented

    def tensor(self):
        a=tf.random_shuffle(self.fixed_latent, seed=None, name=None)
        a=tf.slice(a,[0,0],[self.batch_size,self.shape])
        return a+ self.variance*tf.random_normal(shape=(self.batch_size, self.shape))

