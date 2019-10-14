# Optimal-Transport-GAN

This repository is a Tensorflow implementation of [Training Generative Networks with general Optimal Transport distances](https://arxiv.org/abs/1910.00535). It can be used for MNIST, FASHION-MNIST and CIFAR10.


### Installing

We recommend using conda
```
conda create -n myenv python=3.6
conda activate myenv
pip install -r requirements.txt
```

### Training
We have two methods of training. In the first one we sample ~10 times the amount of generated points
compared to the total amount of real samples for each update step. It is the method that we 
used for the results in the paper. It is more computationally expensive but it ensures 
that no mode collapse occurs. The second method is closer to the standard 
GAN training, in the sense that we only take small batches of generated 
points in every iteration (we still have to go through all the reals).
It is much faster, but like in the GAN training methods, we lose control
of the node dropping.

To train our model with the first method run:
```
python3 Train.py
```

To train our model with the second method run:
```
python3 Simpletrain.py
```

### Latent Space
We used two different ways for sampling points in the latent 
space. The first one is the standard by now option of one Gaussian. As a 
second method, we tried the option of multiple Gaussians  with some 
fixed variance sigma. 

If one picks the standard way for sampling the latent space, will notice that
training with the first method (Train.py) takes much longer (up to 100 times more main steps)
and the level of detail in the generated images seems to be bounded. 
With the second method for training (Simpletrain.py), more 
detail in the results is achieved, but we expect that some node dropping will take
place. We believe that trying to find a model to fit ALL the real data points is 
very hard. Therefore the training has to pick between accuracy and avoiding 
mode collapse. 

It the case of multiple Gaussians, if one makes the number of Gaussians 
too big and  variance small then we have a "perfect fit", or an overfit depending on 
how one sees it. This is the case where we get the best degree of 
details in our Fashion dataset without any node dropping. 
If the number of Gaussians is small then 
it is harder to train and one has to increase the number of critic 
steps (around 10), and variance to avoid 
mode collapse. However, we expect that if the dataset allows it, then 
a small number of Gaussians will result in some clustering of the data. 
We also prefer this method from the standard one, because we believe 
that quite often the data manifold has different  geometry in different parts.


In the case of Fashion Mnist one needs more Gaussians with small variance (<0.4)
compared to the dataset to achieve the best possible approximation. 
We believe that this happens because the Mnist databases do not really 
have any accumulation points, at least not with the Euclidean distance. 
If one takes a smaller number of Gaussians (80%-90% of the points) with medium 
variance (0.5), we observe some interesting phenomena. The training 
method will alternate between dropping some of the points (some mode collapse) 
and  maintain the accuracy (we hypothesize that something similar happens with
WGAN-GP way of training). Further increase in the variance will result in a
training similar to the case of the one Gaussian. 

A more dynamic way of changing the number of gaussian and assigning 
individual variances may result in better data representation. Also some
targeted drop of nodes may result in better fits.

### Experimental Training

In this folder, we keep files with experimental methods of training.
We would like to draw the attention to the critic_first file because it 
is a "proof" that you can train the critic first and then the generator 
(no iterating between the two). However, this takes a lot of time (depending
 on the complexity of the generated points) and is  not optimal. If the
 generated points have some structure (not random noise) already,
 then the critic trains easier
 
 We are welcoming any new ideas for training, and also we are very 
 interested in testing other cost functions that are more suitable for different
 data sets.

### Known issues.

We recently noticed that our code does not function well with a GTX 980 TI.
Very often the generator collapses and gives nan values. We note that
had no issues whatsoever with GTX 960, GTX 1050TI, and Titan X.