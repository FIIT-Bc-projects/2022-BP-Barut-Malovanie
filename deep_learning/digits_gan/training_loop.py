import numpy as np
import torch
import sys

from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from models import Generator, Discriminator

sys.path.append('../utils')
from helpers import init_normal_weights, visualize_images, save_models, visualize_loss, \
    visualize_disc_confidence, create_saved_folder, determine_device

real_label = 1.
fake_label = 0.
img_list = []
G_losses = []
D_losses = []
D_x_arr = []
D_G_z1_arr = []
num_epochs = 100
checkpoints = np.linspace(0, num_epochs, 5, endpoint=False, dtype=np.integer)

create_saved_folder()
device = determine_device(1)
print(f"Using device: {device}")

training_data = datasets.MNIST(root="digits", train=True, download=True, transform=ToTensor())
train_dataloader = DataLoader(training_data, batch_size=128, shuffle=True)
train_images, _ = next(iter(train_dataloader))
visualize_images(train_images[:64], "Real Images", save=True, save_path="saved/real_images_visualization")

generator = Generator().to(device)
generator.apply(init_normal_weights)
print(generator)

discriminator = Discriminator().to(device)
discriminator.apply(init_normal_weights)
print(discriminator)

criterion = nn.BCELoss()
fixed_noise = torch.randn(8, 100, 1, 1, device=device)

optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(generator.parameters(),  lr=0.0002, betas=(0.5, 0.999))

start_time = datetime.now()
print("Starting Training Loop...")
for epoch in range(num_epochs):

    if epoch != 0 and epoch in checkpoints:
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
            for image in fake:
                img_list.append(image)
        save_models(generator, discriminator, epoch)

    for i, data in enumerate(train_dataloader, 0):

        discriminator.zero_grad()
        real_images = data[0].to(device)
        b_size = real_images.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = discriminator(real_images).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        laten_vec = torch.randn(b_size, 100, 1, 1, device=device)
        fake_images = generator(laten_vec)
        label.fill_(fake_label)
        output = discriminator(fake_images.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake_images).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        if i % 50 == 0:
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            D_x_arr.append(D_x)
            D_G_z1_arr.append(D_G_z1)
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f'
                  % (epoch + 1, num_epochs, i, len(train_dataloader), errD.item(), errG.item(), D_x, D_G_z1))

print(f"Training finished Total duration: {datetime.now() - start_time}")

save_models(generator, discriminator, "final")

with torch.no_grad():
    latent_vec = torch.randn(8 * 8, 100, 1, 1, device=device)
    fake = generator(latent_vec).detach().cpu()
    visualize_images(fake, "Generated Images", save=True, save_path="saved/final_images")
    fake = generator(fixed_noise).detach().cpu()
    for image in fake:
        img_list.append(image)

visualize_images(img_list, "Training progression", save=True, save_path="saved/progress")
visualize_loss(G_losses, D_losses, save=True)
visualize_disc_confidence(D_x_arr, D_G_z1_arr, save=True)

