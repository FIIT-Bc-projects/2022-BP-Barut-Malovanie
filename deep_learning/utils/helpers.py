import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from torch import nn


def determine_device(ngpu=0):
    return torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def create_saved_folder(path="./"):
    if not os.path.isdir(path + "saved"):
        os.makedirs(path + "saved")


def init_normal_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def visualize_images(images, title,  save=False, save_path="saved/images", n_row=8):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(images, n_row, padding=2, normalize=True).cpu(), (1, 2, 0)))
    if save:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def save_models(generator, discriminator, progress_number, path="saved"):
    torch.save(generator.state_dict(), path + "/generator_" + str(progress_number))
    torch.save(discriminator.state_dict(), path + "/discriminator_" + str(progress_number))


def visualize_loss(gen_loss, disc_loss, cls_loss=None, save=True, save_path="saved/loss_plot"):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gen_loss, label="Gen")
    plt.plot(disc_loss, label="Disc")
    if cls_loss is not None:
        plt.plot(cls_loss, label="Class")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if save:
        plt.savefig(save_path)
    plt.close()


def visualize_disc_confidence(D_x, D_G_z, save=True, save_path="saved/conf_plot"):
    plt.plot(D_x, label='D(x)')
    plt.plot(D_G_z, label='D(G(z))')
    plt.plot([0.5 for x in range(len(D_x))], label='Ideal')
    plt.xlabel("Iterations")
    plt.ylabel("Prediction")
    plt.legend()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def visualize_disc_acc(acc, save=True, save_path="saved/conf_plot"):
    plt.plot(acc)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
