import os

import matplotlib.pyplot as plt
import numpy as np

from logger.LoggerNew import LoggerNew
from logger.Plotting import make_image_grid
from logger.SummaryType import SummaryType


class LoggerFacade:
    """
    A Facade to give an easy interface to logging data to the tensorboard logs
    """

    def __init__(self, session, writer, dataset_shape):
        self.session = session
        self.writer = writer
        self.logger = LoggerNew(session, writer)
        self.dataset_shape = dataset_shape

    def log_image_grid_fixed(self, first_rows, second_rows, n_iterations, name="real_and_fake_images",
                             save_image=False, ):
        """
        logs the given points as a grid to tensorboard

        :param first_rows: first rows to log
        :param second_rows: second rows to log
        :param n_iterations:  At what iteration should it be saved to tensorboard
        :return: None
        """

        first_grid = make_image_grid(np.reshape(first_rows, (-1,) + self.dataset_shape))
        second_grid = make_image_grid(np.reshape(second_rows, (-1,) + self.dataset_shape))
        combined_grid = np.concatenate((second_grid, first_grid), axis=0)
        # tell tensorboard this is just a single image
        combined_grid = np.expand_dims(combined_grid, axis=0)
        self.logger.write_to_log(combined_grid, name, n_iterations, SummaryType.IMAGE)
        if save_image:
            self._to_image(combined_grid.squeeze(), n_iterations)

    def _to_image(self, matrix, n_iteration, name="fakes"):
        plt.imshow(matrix)
        plt.savefig(self.logger.writer.get_logdir() + os.sep + name + "_" + str(n_iteration))
