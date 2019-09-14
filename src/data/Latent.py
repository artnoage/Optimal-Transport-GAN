from abc import ABC,abstractmethod
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


class GaussianLatent(Latent):
    """
    Latent space where the samples are drawn from a multivariate Gaussian.
    If dim(shape) > 1, samples are drawn from a multivariate Gaussian in np.prod(shape) dimensions
    and then reshaped.
    """
    def __init__(self, shape, batch_size, mean=None, cov=None):
        """
        :param shape: See Latent class.
        :param batch_size: See Latent class.
        :param mean: Mean vector of the multivariate Gaussian with mean.size = np.prod(shape).
            If not supplied, a mean vector is sampled uniformly from the unit cube.
        :param cov: Covariance matrix of the multivariate Gaussian with cov.shape = np.prod(shape)^2.
            If an integer is supplied, the covariance matrix is cov * I, where I is the identity matrix.
            If not supplied, the covariance matrix is 0.25 * I.
        """
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size

        # Sample a flattened array and reshape afterwards.
        size = np.prod(self.shape)

        if mean is None:
            self.mean = np.random.uniform(0.5, 0.5, size)
        else:
            self.mean = mean

        if cov is None:
            cov = 0.9

        if isinstance(cov, float):
            self.cov = cov * np.identity(size)
        else:
            self.cov = cov

    def sample(self, batch_size=None):
        """
        See Latent.sample().
        :param batch_size: Batch size if not default_batchsize.
        """
        if batch_size is None:
            batch_size = self.batch_size

        samples = np.random.multivariate_normal(self.mean, self.cov, batch_size)

        return np.reshape(samples, (batch_size, self.shape))

class UniformLatent(Latent):

    def __init__(self, shape, batch_size):
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        samples = np.random.uniform(-1, 1, [batch_size, self.shape])
        return samples

class TensorflowLatent(Latent):

    def __init__(self, shape=None, batch_size=None):
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size
        self.name = "Tensorflow_latent"
    def sample(self, batch_size=None):
        raise NotImplemented

    def tensor(self):
        return tf.random_normal(shape=(self.batch_size, self.shape))


class Assignment_latent(Latent):

    def __init__(self, shape=None, batch_size=None):
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size
        self.initial_points=np.random.normal(0,1,(50000,self.shape))
        self.fixed_latent = tf.constant(self.initial_points, dtype=tf.float32)
        self.name="Assignment_latent"
    def sample(self, batch_size=None):
        raise NotImplemented

    def tensor(self):
        a=tf.random_shuffle(self.fixed_latent, seed=None, name=None)
        a=tf.slice(a,[0,0],[self.batch_size,self.shape])
        return a+0.02*tf.random_normal(shape=(self.batch_size, self.shape))

class StaticLatent(Latent):
    # TODO: Implement here latent spaces with PCA etc.
    pass
