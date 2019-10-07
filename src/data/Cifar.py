import matplotlib
import numpy as np
import tensorflow as tf

from data.DatasetABC import DatasetABC

matplotlib.use('Agg')


class Cifar(DatasetABC):
    """
    Class for the cifar dataset.
    """
    def __init__(self, batch_size,dataset_size):
        """
        :param batch_size: The default batch size.
        """
        # Load data.
        super().__init__()
        samples,labels = self.generate_data()
        # Initialize class.
        self.name = "Cifar10"
        self.data = samples
        self.labels = labels
        self.dataset_size = dataset_size
        self.shape = (32, 32, 3)
        self.batch_size = batch_size

    @staticmethod
    def generate_data():
        (data, label), (_, _) = tf.keras.datasets.cifar10.load_data()
        data = data / 255
        data = (data - 0.5) / 0.5
        data = data.reshape(data.shape[0], -1)
        data = np.float32(data)

        return data, label