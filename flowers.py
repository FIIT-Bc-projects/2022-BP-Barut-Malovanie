import os
import shutil

import h5py
import numpy as np
from keras import Sequential
from keras.datasets.mnist import load_data
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose
from keras.optimizers import Adam
from matplotlib import pyplot
from numpy import zeros, ones, vstack, expand_dims
from numpy.random import randn, randint

def generate_latent_points(latent_dim, n):
    x_input = np.random.randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input

def load_data():
    pass