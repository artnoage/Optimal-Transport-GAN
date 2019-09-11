import tensorflow as tf
from data.Dataset import DatasetNew


class Cifar10_32(DatasetNew):
    """
    Class for the cifar10 dataset.
    """

    def __init__(self, batch_size):
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
        self.n_elements = samples.shape[0]
        self.shape = (32, 32, 3)
        self.batch_size = batch_size

    @staticmethod
    def generate_data():
        (data, label), (_, _) = tf.keras.datasets.cifar10.load_data()
        data = data / 255
        data = (data - 0.5) / 0.5
        data = data[:1000]
        label = label[:1000]
        data = data.reshape(data.shape[0], -1)

        return data, label