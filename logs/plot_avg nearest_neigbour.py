
from logger.Plotting import find_average_nearest_neigbour
import numpy as np
import scipy.misc
from models.CostType import CostType
import tensorflow as tf
from logger.Plotting import find_distance_matrix
from logger.Plotting import make_image_grid
import matplotlib
#matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

(data, label), (_, _) = tf.keras.datasets.mnist.load_data()
imgs_32 = [scipy.misc.imresize(data[idx], (32, 32, 1))
           for idx in range(data.shape[0])]
data = np.expand_dims(np.asarray(imgs_32).astype(np.float32), axis=3)
data = data / 255
data = (data - 0.5) / 0.5
data = data.reshape(data.shape[0], -1)
data = data[:5000]
label = label[:5000]

cost_type = CostType.WGAN_GRADIENT_PENALTY
data_fake = np.load("fakes.npy")
if cost_type == CostType.SSIM:
    dist_mat =find_distance_matrix(data+1,data_fake[:40]+1,cost_type)
else:
    dist_mat = find_distance_matrix(data, data_fake[:40], cost_type)
print(dist_mat.shape)
if cost_type == CostType.SSIM:
    idx = np.argmax(dist_mat, axis=0)
else:
    idx = np.argmin(dist_mat, axis=0)
real_idx = idx[:36]
fake_idx = range(40)[:36]
real_images =make_image_grid(data[real_idx],(32,32),6,6)
fake_images =make_image_grid(data_fake[fake_idx],(32,32),6,6)
bar = np.ones((6*32,5))
combined = np.hstack((fake_images,bar,real_images))
plt.imshow(combined,cmap='Greys_r')
plt.show()
plt.imshow(real_images,cmap='Greys_r')
plt.axis('off')
plt.savefig("mnist_gan_real_images", bbox_inches='tight', pad_inches=0)
plt.imshow(fake_images,cmap='Greys_r')
plt.axis('off')
plt.savefig("mnist_gan_fake_images", bbox_inches='tight', pad_inches=0)


