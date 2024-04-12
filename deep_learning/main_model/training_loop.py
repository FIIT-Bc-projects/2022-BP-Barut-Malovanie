import torch
import numpy as np
import sys

from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from datasets import FlowersDataset

sys.path.append('../utils')
from helpers import init_normal_weights, visualize_images, save_models, visualize_loss, \
    visualize_disc_confidence, create_saved_folder, determine_device, visualize_disc_acc
from transforms import exp_2

real_label = 1.
fake_label = 0.
img_list = []
G_losses = []
D_losses = []
D_losses_class = []
D_x_arr = []
D_G_z1_arr = []
D_class_acc = []
num_epochs = 200
batch_size = 64
latent_dim = 100
checkpoints = np.linspace(0, num_epochs, 8, endpoint=False, dtype=np.integer)

create_saved_folder()
device = determine_device(1)
print(f"Using device: {device}")

dataset = FlowersDataset("data/", train=True, device=device, transform=exp_2)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
train_images, class_labels, encoded = next(iter(train_dataloader))
visualize_images(train_images[:64], "Training Images",  save=True, save_path="saved/real_images_visualization")

# dataset_test = FlowersDataset("data/", train=False, device=device, transform=exp_2)
# test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

generator = Generator().to(device)
generator.apply(init_normal_weights)
print(generator)

discriminator = Discriminator().to(device)
discriminator.apply(init_normal_weights)
print(discriminator)

fixed_noise = torch.randn(8, latent_dim, 1, 1, device=device)
encoded_g = torch.reshape(encoded, (*encoded.shape, 1, 1))
fixed_prompts = encoded_g[:8]

criterion_disc = nn.BCELoss()
criterion_class = nn.CrossEntropyLoss()

optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(),  lr=0.0002, betas=(0.5, 0.999))

start_time = datetime.now()
print("Starting Training Loop...")
for epoch in range(num_epochs):

    fake = generator(fixed_noise, fixed_prompts).detach().cpu()
    visualize_images(fake, f"Images at epoch {epoch + 1}")

    if epoch != 0 and epoch in checkpoints:
        with torch.no_grad():
            for image in fake:
                img_list.append(image)
        save_models(generator, discriminator, epoch)

    for i, data in enumerate(train_dataloader):

        real_images = data[0].to(device)
        class_labels = data[1].to(device)
        encoded = torch.reshape(data[2], (*data[2].shape, 1, 1)).to(device)
        one_hot = torch.nn.functional.one_hot(class_labels, 102).float()
        b_size = real_images.size(0)

        #disc train on real image, correct class, correct prompt
        discriminator.zero_grad()
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        noise = torch.rand((b_size, ), device=device)
        label = label - (noise / 5)
        disc, cls = discriminator(real_images, encoded)
        disc = disc.view(-1)
        errD_disc_real = criterion_disc(disc, label)
        errD_disc_real.backward(retain_graph=True)
        errD_class_real = criterion_class(one_hot, cls)
        errD_class_real.backward()
        D_x = disc.mean().item()
        acc = torch.eq(class_labels, cls.max(1).indices).sum()

        #disc train on real image, correct class, different prompt
        shift_encoded = torch.cat((encoded[1:], encoded[0].unsqueeze(0)))
        label.fill_(fake_label)
        noise = torch.rand((b_size, ), device=device)
        label = label + (noise / 3.35)
        disc, _ = discriminator(real_images, shift_encoded)
        disc = disc.view(-1)
        errD_disc_wrong = criterion_disc(disc, label)
        errD_disc_wrong.backward()

        #disc train on fake image, correct class, correct prompt
        latent_vec = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_images = generator(latent_vec, encoded)
        label.fill_(fake_label)
        noise = torch.rand((b_size, ), device=device)
        label = label + (noise / 3.35)
        disc, cls = discriminator(fake_images.detach(), encoded)
        errD_disc_fake = criterion_disc(disc.view(-1), label)
        errD_disc_fake.backward(retain_graph=True)
        errD_class_fake = criterion_class(one_hot, cls)
        errD_class_fake.backward()
        D_G_z1 = disc.mean().item()
        errD = errD_disc_real + errD_disc_fake
        errD_class = errD_class_real + errD_class_fake
        optimizerD.step()
        acc += torch.eq(class_labels, cls.max(1).indices).sum()

        #generator train
        generator.zero_grad()
        label.fill_(real_label)
        disc, cls = discriminator(fake_images, encoded)
        disc = disc.view(-1)
        errG = criterion_disc(disc, label)
        errG.backward(retain_graph=True)
        errG_class = criterion_class(one_hot, cls)
        errG_class.backward()
        optimizerG.step()

        if i % 10 == 0:
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            D_losses_class.append(errD_class.item())
            D_x_arr.append(D_x)
            D_G_z1_arr.append(D_G_z1)
            acc = (acc / (b_size*2)).item()
            D_class_acc.append(acc)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_D_class: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f\tAcc: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataloader), errD.item(), errD_class.item(), errG.item(),
                     D_x, D_G_z1, acc))


print(f"Training finished Total duration: {datetime.now() - start_time}")

save_models(generator, discriminator, "final")

with torch.no_grad():
    latent_vec = torch.randn(encoded_g.size(0), latent_dim, 1, 1, device=device)
    fake = generator(latent_vec, encoded_g).cpu()
    visualize_images(fake, "Final generated images", save=True, save_path="saved/final_images")
    fake = generator(fixed_noise, fixed_prompts).detach().cpu()
    for image in fake:
        img_list.append(image)

visualize_images(img_list, "Training progress", save=True, save_path="saved/progress", n_row=8)
visualize_loss(G_losses, D_losses, D_losses_class, save_path="saved/loss_plot")
visualize_disc_confidence(D_x_arr, D_G_z1_arr,  save_path="saved/conf_plot")
visualize_disc_acc(D_class_acc, save_path="saved/acc_plot")

