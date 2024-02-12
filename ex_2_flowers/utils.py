import numpy as np
import matplotlib
from matplotlib import pyplot
from numpy.random import randn
import os
import tensorflow as tf
#matplotlib.use('TkAgg')


def save_loss_plot(disc_loss, gen_loss, path):
    pyplot.plot(disc_loss, label='discriminator')
    pyplot.plot(gen_loss, label='generator')
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.savefig('saved/' + path)
    pyplot.close()


def save_acc_plot(acc_real, acc_fake, path):
    pyplot.plot(acc_real, label='accuracy real')
    pyplot.plot(acc_fake, label='accuracy fake')
    pyplot.plot([0.5 for x in range(len(acc_real))], label='ideal accuracy')
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Accuracy")
    pyplot.legend()
    pyplot.savefig('saved/' + path)
    pyplot.close()


def crate_saved_dir():
    if not os.path.isdir("saved/"):
        os.makedirs("saved/")


def mount_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(physical_devices))
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def visualize_images(images, shape, save=False, save_path=None):
    n_images = np.shape(images)[0]
    for i in range(n_images):
        pyplot.subplot(*shape, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(images[i])
    if save:
        pyplot.savefig("saved/" + save_path, bbox_inches='tight')


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
    X = g_model.predict([inputs, latent_vec], verbose=0)
    y = np.zeros((n_samples, 1))
    return X, y

