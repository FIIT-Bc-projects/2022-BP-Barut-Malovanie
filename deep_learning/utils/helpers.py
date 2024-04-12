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


def running_average(arr, window=10):
    average_data = []
    for i in range(len(arr) - window + 1):
        average_data.append(np.mean(arr[i:i + window]))
    for ind in range(window - 1):
        average_data.insert(0, np.nan)
    return average_data


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
    ax = plt.gca()
    ax.set_ylim([0, 10])
    gen_loss_avg = running_average(gen_loss)
    disc_loss_avg = running_average(disc_loss)
    plt.title("Generator and Discriminator Loss")
    plt.plot(gen_loss, alpha=0.2, color="tab:blue")
    plt.plot(disc_loss, alpha=0.2, color="tab:green")
    plt.plot(gen_loss_avg, label="Gen", alpha=0.9, color="tab:blue")
    plt.plot(disc_loss_avg, label="Disc", alpha=0.9, color="tab:green")
    if cls_loss is not None:
        cls_loss_avg = running_average(cls_loss)
        plt.plot(cls_loss, alpha=0.2, color="tab:orange")
        plt.plot(cls_loss_avg, label="Class", alpha=0.9, color="tab:orange")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    if save:
        plt.savefig(save_path)
    plt.close()


def visualize_disc_confidence(D_x, D_G_z, save=True, save_path="saved/conf_plot"):
    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Predictions")
    D_x_avg = running_average(D_x)
    D_G_z_avg = running_average(D_G_z)
    plt.plot(D_x, color="tab:blue", alpha=0.2)
    plt.plot(D_G_z, color="tab:orange", alpha=0.2)
    plt.plot(D_x_avg, label='D(x)', color="tab:blue", alpha=0.9)
    plt.plot(D_G_z_avg, label='D(G(z))', color="tab:orange", alpha=0.9)
    plt.plot([0.5 for x in range(len(D_x))], alpha=0.5, label='Ideal', color="tab:green")
    plt.xlabel("Iterations")
    plt.ylabel("Prediction")
    plt.legend()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


def visualize_disc_acc(acc, save=True, save_path="saved/conf_plot"):
    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Classification Accuracy")
    acc_avg = running_average(acc)
    plt.plot(acc, alpha=0.2, color="tab:orange")
    plt.plot(acc_avg, alpha=0.9, color="tab:orange")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    if save:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
