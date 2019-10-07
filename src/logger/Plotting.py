import os

import imageio
import matplotlib
import numpy as np
import ot

matplotlib.use('Agg')


def make_image_grid(image_array, to_shape=None, rows=6, columns=3, save_image=False, name=''):
    size = image_array.shape[0]
    position = 0
    image_list = []
    for image_number in range(rows * columns):
        image_idx = position % size
        if to_shape is None:
            image_list.append(image_array[image_idx])
        else:
            image_list.append(np.reshape(image_array[image_idx], to_shape))
        position += 1
    slide_list = []
    for column in range(columns):
        slide_list.append(np.concatenate(image_list[column * rows:(column + 1) * rows], axis=1))
    full = np.vstack(slide_list)
    if save_image:
        if save_image:
            os.makedirs(os.path.join(os.getcwd(), "pictures"), exist_ok=True)
            path = os.path.join(os.getcwd(), "pictures", name + '.png')
            imageio.imwrite(path, full)
    return full


def wasserstein_distance(X, Y, Npoints_arg):
    X = np.reshape(X, (Npoints_arg, -1))

    Y = np.reshape(Y, (Npoints_arg, -1))
    M = ot.dist(X, Y, metric='euclidean')
    n = Npoints_arg
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    G0 = ot.emd(a, b, M, numItermax=200000)
    return np.sum(G0 * M)

