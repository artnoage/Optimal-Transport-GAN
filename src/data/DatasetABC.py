import numpy as np
from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
import tensorflow as tf


class DatasetABC(ABC):
    """
    An abstract class for different datasets.

    Subclasses need to implement the constructor where all class variables are set.

    The sample methods are implemented here and can be inherited.
    """

    def __init__(self):
        """
        :param name: Human-readable name for the dataset.
        :param data: Actual data. The first dimension accounts for the different samples.
        :param n_elem: Number of elements in data.
        :param shape: The shape of a *single* sample.
        :param batch_size: The default batch size.
        """
        self.name = None
        self.data = None
        self.labels = None
        self.dataset_size = None
        self.shape = None
        self.batch_size = None
        self.dataset_iterator = None

    @staticmethod
    @abstractmethod
    def generate_data():
        """
        Define the base mathod to generate subclass

        :return: subclass and label array
        """
        pass

    def sample_idx(self, label=None, batch_size=None):
        """
        Sample indices from dataset.

        This is the most generic way to sample from a dataset, both traindata and labels::

            idx = dataset.sample_idx()
            subclass = dataset.traindata[idx, :]
            label = dataset.trainlabel[idx]

        :param label: Only return indices with given label.
        :param batch_size: Batch size if none takes the predefined batch size.

        :return: A np.array containing indices.
        """
        if batch_size is None:
            batch_size = self.batch_size

        if label is not None:
            positions = np.where(self.labels == label)[0]
            idx = np.random.choice(positions, batch_size, replace=False)
        else:
            idx = np.random.choice(self.dataset_size, batch_size, replace=False)

        return idx

    def sample(self, label=None, batch_size=None):
        """
        Directly sample subclass from dataset. This is a shortcut for sample_idx().

        :param label: Only return indices with given label.
        :param batch_size: Batch size if none takes the predefined batch size.

        :return: A np.array of shape (batchsize,) + dataset.shape.
        """
        idx = self.sample_idx(label=label, batch_size=batch_size)

        return self.data[idx, :]

    def get_iterator(self):
        """
        Returns an iterator over the data

        :return: an iterator over the data
        """
        if self.dataset_iterator is None:
            dataset = tf.data.Dataset.from_tensor_slices(self.data.astype(np.float32)).repeat().batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            self.dataset_iterator = iterator
            return iterator.get_next()
        else:
            return self.dataset_iterator.get_next()

    def get_total_shape(self):
        """
        Returns the total dimensions of the dataset

        :return: the total dimension size
        """
        return reduce(mul, self.shape)
