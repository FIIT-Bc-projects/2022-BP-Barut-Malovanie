import torch
import sys
from datasets import FlowersDataset
from torch.utils.data import DataLoader
from torchmetrics.image import InceptionScore, FrechetInceptionDistance
from models import Generator
sys.path.append('../utils')
from helpers import determine_device
from transforms import scale_for_metrics

device = determine_device(1)
print('Using device:', device)
batch_size = 256

dataset = FlowersDataset("data/", train=True, device=device, transform=scale_for_metrics)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataset_test = FlowersDataset("data/", train=False, device=device, transform=scale_for_metrics)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)


generator = Generator().to(device)
state_dict = torch.load('../../trained_models/flowers_generator_final', map_location=device)
generator.load_state_dict(state_dict)
generator.eval()


def calculate_metrics(dataloader, gen):
    inception = InceptionScore(normalize=True)
    fid = FrechetInceptionDistance(normalize=True)

    for data in dataloader:
        real_images = data[0].to(device)
        encoded = torch.reshape(data[2], (*data[2].shape, 1, 1)).to(device)
        b_size = real_images.size(0)

        latent_vec = torch.randn(b_size, 100, 1, 1, device=device)
        fake_images = generator(latent_vec, encoded)
        fake_images = (fake_images + 1) / 2

        real_images = real_images.to("cpu")
        fake_images = fake_images.to("cpu")
        inception.update(fake_images)

        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

    return inception.compute(), fid.compute()


train_is, train_fid = calculate_metrics(train_dataloader, generator)
test_is, test_fid = calculate_metrics(test_dataloader, generator)

print("Train Metrics\n \tInception Score mean: %.4f, \tInception Score std: %.4f\n "
      "\tFID mean: %.4f" % (train_is[0].item(), train_is[1].item(), train_fid.item()))
print("Test Metrics\n \tInception Score mean: %.4f, \tInception Score std: %.4f\n "
      "\tFID mean: %.4f" % (test_is[0].item(), test_is[1].item(), test_fid.item()))




