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
of mode collapse.

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
second method, we tried using multiple Gaussians  with some 
fixed variance sigma. 

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
