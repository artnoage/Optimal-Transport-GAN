import tensorflow as tf
import numpy as np
import sklearn.metrics

import matplotlib

from data.Dataset import DatasetNew

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.misc

class Fashion32(DatasetNew):
    """
    Class for the Fashion32 dataset.
    """
    def __init__(self, batch_size,dataset_size):
        """
        :param batch_size: The default batch size.
        """
        # Load data.
        super().__init__()
        self.shape = (32, 32, 1)
        data,labels = self.generate_data()
        # Initialize class.
        self.name = "Fashion32"
        self.data = data
        self.labels = labels
        self.dataset_size= dataset_size

        self.batch_size = batch_size

    @staticmethod
    def generate_data():
        shape = (32, 32, 1)
        (data, label), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        imgs_32 = [scipy.misc.imresize(data[idx], shape)
                   for idx in range(data.shape[0])]
        data = np.expand_dims(np.asarray(imgs_32).astype(np.float32), axis=3)
        data = data / 255
        data = (data - 0.5) / 0.5
        data = data.reshape(data.shape[0], -1)
        return data, label


