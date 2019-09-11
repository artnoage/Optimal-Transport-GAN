import io
import os
import pickle
from collections.__init__ import Counter

import imageio
import numpy as np
import ot
import scipy as scipy  # Ensure PIL is also installed: pip install pillow # TODO: Why?

import matplotlib
import skimage

from models.CostType import CostType

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import scipy.misc
import tensorflow as tf



frame_index = [0]


def generate_density(true_samples, save_image=False, name=''):
    """
    Plots a density plot by counting the number of samples that fall into a region and plotting the contours
    of the resulting values. Returns the plot as an array so it can be saves as a image to tensorboard

    :param true_samples: input data
    :param save_image: if an image of the plot should be saved
    :param name: image will be saved to to os.getcwd() +  "pictures" + name '.png'
    :return: Array representation of the image
    """
    H, xedges, yedges = np.histogram2d(true_samples[:, 0], true_samples[:, 1], (100, 100), [[-1, 1], [-1, 1]])
    x_2d, y_2d = np.meshgrid(xedges[:-1], yedges[:-1])
    plt.contourf(y_2d, x_2d, H, cmap=plt.cm.Greys)
    #plt.scatter(true_samples[:, 0],true_samples[:, 1],s=1)
    if save_image:
        os.makedirs(os.path.join(os.getcwd(), "pictures"), exist_ok=True)
        path = os.path.join(os.getcwd(), "pictures",name + '.png')
        plt.savefig(path)  # saving the plotting result on a picture each 100 iteration
        print(path)
    return _image_to_array()


def plot_fake_and_real(gen_samples, true_samples, testname="", save_image=False, collab_add=""):
    """
    Plots generated samples as well as true samples in the same matplot lib figure. Returns the array representation
    of the picture.

    :param gen_samples: generator samples
    :param true_samples: real samples
    :param testname: naming the file if it get saved : os.getcwd(), collab_add, testname, "pictures"
    :param save_image: if the image should be saved
    :param collab_add: additional option to change the location / name of the image e.g. to porperly save the image in collab
    :return: array representation of the matplotlib figure
    """
    plt.scatter(true_samples[:, 0], true_samples[:, 1], c='green', marker='+', alpha=0.5,label='data')  # plotting the real data
    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], c='red', marker='+', alpha=.7,label='gen')  # plotting the generated data
    plt.xlim([-2, 2])  # determine figure x range
    plt.ylim([-2, 2])  # determine figure x range
    plt.legend(loc=1)
    if save_image:
        os.makedirs(os.path.join(os.getcwd(), collab_add, testname, "pictures"), exist_ok=True)
        plt.savefig(os.path.join(os.getcwd(), collab_add, testname, "pictures", 'frame' + str(
            frame_index[0]) + '.png'))  # saving the plotting result on a picture each 100 iteration
        frame_index[0] += 1
    return _image_to_array()


def _image_to_array():
    """
    Transforms the current matplotlib plot to an array

    :return: the array
    """
    # write the plot to a buffer. We want to return a numpy array.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    im = scipy.misc.imread(buf)
    buf.close()
    plt.close()
    plt.clf()
    return np.expand_dims(im, axis=0)


def plot_assignments(reals, fakes, real_label,dataset,path):
    n_classes = 10
    colors = cm.rainbow(np.linspace(0, 1, n_classes))
    reals = np.vstack(np.array(reals))[0:5000]
    fakes = np.vstack(np.array(fakes))[0:5000]
    real_label = np.array(real_label).flatten()[0:5000]
    plt.figure(figsize=(20, 10))
    data_real = dataset.data
    data_real_label = dataset.labels
    for cls in range(n_classes):
        mask = np.where(real_label == cls)
        data_mask = np.where(data_real_label == cls)
        plt.scatter(data_real[data_mask, 0], data_real[data_mask, 1], s=6, color=colors[cls])
        plt.scatter(fakes[mask, 0], fakes[mask, 1], s=18, color=colors[cls],edgecolors='black',linewidths=0.2)
    for i in range(0, len(fakes)):
        label = real_label[i]
        plt.plot((reals[i, 0], fakes[i, 0]), (reals[i, 1], fakes[i, 1]),
                 alpha=0.1, color=colors[label])
    unique, counts = np.unique(real_label, return_counts=True)
    patches = []
    for i in range(len(unique)):
        patches.append(mpatches.Patch(color=colors[unique[i]], label=str(counts[i])))
    plt.legend(handles=patches, bbox_to_anchor=(1.00, 1), loc=2, borderaxespad=0.)
    plt.savefig(path)
    plt.close()

def plot_histogram(reals,real_idx,path,num_bins,save=False):
    unique ,counts = np.unique(real_idx,axis=0,return_counts=True)
    print(np.hstack((unique, counts)).shape)
    #if save :
    #    np.save(path+str("assign_matrix"),np.vstack((unique,counts)))
    #plt.bar(unique,counts)
    #plt.savefig(path)
    #plt.close()


def make_image_grid(image_array,to_shape=None,rows=6,columns=3,save_image=False,name=''):
    size = image_array.shape[0]
    position = 0
    image_list = []
    for image_number in range(rows*columns):
        image_idx = position % size
        if to_shape is None:
            image_list.append(image_array[image_idx])
        else:
            image_list.append(np.reshape(image_array[image_idx],to_shape))
        position+=1
    slide_list= []
    for column in range(columns):
        slide_list.append(np.concatenate(image_list[column*rows:(column+1)*rows],axis=1))
    full = np.vstack(slide_list)
    if save_image:
        if save_image:
            os.makedirs(os.path.join(os.getcwd(), "pictures"), exist_ok=True)
            path = os.path.join(os.getcwd(), "pictures", name + '.png')
            imageio.imwrite(path, full)
    return full


def wasserstein_distance(X,Y,Npoints_arg):
    X=np.reshape(X,(Npoints_arg,-1))

    Y = np.reshape(Y, (Npoints_arg, -1))
    M = ot.dist(X, Y, metric='euclidean')
    n = Npoints_arg
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    G0 = ot.emd(a, b, M, numItermax=200000)
    return np.sum(G0 * M)


def save_variable(var, name,path):
    filename = name
    outfile = open(path+os.sep+filename, 'wb+')
    pickle.dump(var, outfile)
    outfile.close()


def print_assignment(session,assignment,dataset_facade,dataset):
    jan = Counter(session.run(assignment, dataset_facade.get_data_dict()))
    jan2 = np.zeros(dataset.n_elem)
    for i in jan.keys():
        jan2[i] = jan.get(i)
    print(jan2)
    print(Counter(jan2).get(0))

def find_average_nearest_neigbour(real_points, fake_points, cost_type,norm_mat = None):
    if cost_type == CostType.WGAN_GRADIENT_PENALTY:
        dist_mat = scipy.spatial.distance.cdist(real_points,fake_points,metric="euclidean")
    elif cost_type == CostType.WGAN_WEIGHT_CLIPPING:
        dist_mat = scipy.spatial.distance.cdist(real_points, fake_points, metric="euclidean")
    elif cost_type == CostType.WGAN_WEIGTHED_NORM:
        dist_mat = scipy.spatial.distance.cdist(real_points, fake_points, metric="euclidean",w=norm_mat)
    elif cost_type == CostType.WGAN_COSINE:
        dist_mat = scipy.spatial.distance.cdist(real_points, fake_points, metric="cosine")
    elif cost_type == CostType.WGAN_SQUARE:
        dist_mat = scipy.spatial.distance.cdist(real_points, fake_points, metric="sqeuclidean")
    elif cost_type == CostType.SSIM:
        f = skimage.measure.compare_ssim
        dist_mat = scipy.spatial.distance.cdist(real_points, fake_points,f)

    else:
        raise ValueError("This gan has no measurement for closest neighbour")

    idx = np.argmin(dist_mat, axis=0)
    unique, counts = np.unique(idx, return_counts=True)
    return np.average(counts)

def get_inception_scores(images, batch_size, num_inception_images):
  """Get Inception score for some images.
  Args:
    images: Image minibatch. Shape [batch size, width, height, channels]. Values
      are in [-1, 1].
    batch_size: Python integer. Batch dimension.
    num_inception_images: Number of images to run through Inception at once.
  Returns:
    Inception scores. Tensor shape is [batch size].
  Raises:
    ValueError: If `batch_size` is incompatible with the first dimension of
      `images`.
    ValueError: If `batch_size` isn't divisible by `num_inception_images`.
  """
  tfgan = tf.contrib.gan
  # if we get a vector rep of the image we need to transform it
  if len(images.shape) == 2:

      images = tf.reshape(images,shape=(batch_size,32,32,3))
  # Validate inputs.
  images.shape[0:1].assert_is_compatible_with([batch_size])
  if batch_size % num_inception_images != 0:
    raise ValueError(
        '`batch_size` must be divisible by `num_inception_images`.')

  # Resize images.
  size = 299
  resized_images = tf.image.resize_bilinear(images, [size, size])

  # Run images through Inception.
  num_batches = batch_size // num_inception_images
  inc_score = tfgan.eval.inception_score(
      resized_images, num_batches=num_batches)

  return inc_score
