import numpy as np
from matplotlib import pyplot
from numpy.random import randn
import os
import tensorflow as tf


def crate_saved_dir():
    if not os.path.isdir("saved/"):
        os.makedirs("saved/")


def mount_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def visualize_images(images, shape, save_path):
    n_images = np.shape(images)[0]
    for i in range(n_images):
        pyplot.subplot(*shape, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(images[i])
    pyplot.savefig("saved/" + save_path, bbox_inches='tight')
    pyplot.close()


def generate_latent_points(latent_dim, n):
    x_input = np.random.randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input


def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y


def generate_con_fake_samples(g_model, inputs, latent_dim, n_samples):
    if np.shape(inputs)[0] != n_samples:
        return []
    latent_vec = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict([inputs, latent_vec])
    y = np.zeros((n_samples, 1))
    return X, y
