import torch
import sys
import os
import subprocess
import shutil

from datasets import FlowersDatasetRaw
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models import Generator, Encoder
sys.path.append('../utils')
from helpers import determine_device
from transforms import scale_for_metrics

device = determine_device(1)
print('Using device:', device)
batch_size = 256


def calculate_clip_score(images, prompts):
    os.makedirs("tmp", exist_ok=True)
    os.makedirs("tmp/images", exist_ok=True)
    os.makedirs("tmp/prompts", exist_ok=True)

    i = 1
    for image, prompt in zip(images, prompts):
        save_image(image, "tmp/images/" + str(i) + ".png")
        with open('tmp/prompts/' + str(i) + '.txt', 'w') as f:
            f.write(prompt)
        i += 1

    result = subprocess.run(['python', '-m', 'clip_score', 'tmp/images', 'tmp/prompts'], capture_output=True, text=True)
    print(result.stdout)
    shutil.rmtree("tmp")


dataset = FlowersDatasetRaw("data/", train=True, device=device, transform=scale_for_metrics)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
images, _, prompts, _ = next(iter(train_dataloader))

print("Training Data:")
calculate_clip_score(images, prompts)

generator = Generator().to(device)
state_dict = torch.load('saved/generator_final', map_location=device)
generator.load_state_dict(state_dict)
generator.eval()

encoded_prompts = Encoder.encode_prompt(list(prompts), device=device)
encoded_prompts = torch.reshape(encoded_prompts, (*encoded_prompts.shape, 1, 1))
latent_vec = torch.randn(images.shape[0], 100, 1, 1, device=device)
fake_images = generator(latent_vec, encoded_prompts)
fake_images = (fake_images + 1) / 2
fake_images = fake_images.cpu()
print("Generator training:")
calculate_clip_score(fake_images, prompts)


dataset_test = FlowersDatasetRaw("data/", train=False, device=device, transform=scale_for_metrics)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
images, _, prompts, _ = next(iter(test_dataloader))

print("Test Data:")
calculate_clip_score(images, prompts)

encoded_prompts = Encoder.encode_prompt(list(prompts), device=device)
encoded_prompts = torch.reshape(encoded_prompts, (*encoded_prompts.shape, 1, 1))
latent_vec = torch.randn(images.shape[0], 100, 1, 1, device=device)
fake_images = generator(latent_vec, encoded_prompts)
fake_images = (fake_images + 1) / 2
fake_images = fake_images.cpu()
print("Generator test")
calculate_clip_score(fake_images, prompts)
