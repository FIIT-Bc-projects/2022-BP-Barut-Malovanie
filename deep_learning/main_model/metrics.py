import torch
import sys
import clip

from datasets import FlowersDataset
from statistics import mean
from torch.utils.data import DataLoader
from torchmetrics.image import InceptionScore, FrechetInceptionDistance
from models import Generator

sys.path.append('../utils')
from helpers import determine_device
from transforms import scale_for_metrics, clip_img_transform

device = determine_device(1)
print('Using device:', device)
batch_size = 64


class CLIPScore:
    def __init__(self, device="cpu"):
        model, _ = clip.load('ViT-B/32')
        self.model = model.to(device)
        self.model.eval()
        self.values = []
        self.i = 0

    def update(self, images, text_features):
        images = clip_img_transform(images)
        with torch.no_grad():
            image_features = self.model.encode_image(images)
        text_features = text_features.type(image_features.dtype)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = torch.reshape(image_features, (image_features.shape[0], 1, -1))
        product = (image_features @ text_features[..., None])[..., 0]
        self.values.append(product.mean().item())
        self.i += 1

    def compute(self):
        return mean(self.values) / self.i


dataset = FlowersDataset("data/", train=True, device=device, transform=scale_for_metrics)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

dataset_test = FlowersDataset("data/", train=False, device=device, transform=scale_for_metrics)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

generator = Generator().to(device)
state_dict = torch.load('../../trained_models/flowers_generator_final', map_location=device)
generator.load_state_dict(state_dict)
generator.eval()


def calculate_metrics(dataloader):
    inception_gen = InceptionScore(normalize=True)
    inception_data = InceptionScore(normalize=True)
    clip_score_gen = CLIPScore(device)
    clip_score_data = CLIPScore(device)
    fid = FrechetInceptionDistance(normalize=True)

    for data in dataloader:
        real_images = data[0].to(device)
        encoded_prompts = data[2].to(device)
        encoded_shaped = torch.reshape(encoded_prompts, (*encoded_prompts.shape, 1, 1))
        b_size = real_images.size(0)

        latent_vec = torch.randn(b_size, 100, 1, 1, device=device)
        fake_images = generator(latent_vec, encoded_shaped)
        fake_images = (fake_images + 1) / 2

        clip_score_data.update(real_images, encoded_prompts)
        clip_score_gen.update(fake_images, encoded_prompts)

        real_images = real_images.to("cpu")
        fake_images = fake_images.to("cpu")
        inception_gen.update(fake_images)
        inception_data.update(real_images)

        fid.update(real_images, real=True)
        fid.update(fake_images, real=False)

    return (inception_gen.compute(), fid.compute(), clip_score_gen.compute(),
            inception_data.compute(), clip_score_data.compute())


train_is, train_fid, train_clip, train_data_is, train_data_clip = calculate_metrics(train_dataloader)
test_is, test_fid, test_clip, test_data_is, test_data_clip = calculate_metrics(test_dataloader)

print("Train Data Metrics\n \tInception Score mean: %.4f, \tInception Score std: %.4f\n " "\tCLIP Score: %.4f"
      % (train_data_is[0].item(), train_data_is[1].item(), train_data_clip))
print("Test Data Metrics\n \tInception Score mean: %.4f, \tInception Score std: %.4f\n " "\tCLIP Score: %.4f"
      % (test_data_is[0].item(), test_data_is[1].item(), test_data_clip))
print("Train Metrics\n \tInception Score mean: %.4f, \tInception Score std: %.4f\n "
      "\tFID: %.4f \tTrain CLIP Score: %.4f" % (train_is[0].item(), train_is[1].item(), train_fid.item(), train_clip))
print("Test Metrics\n \tInception Score mean: %.4f, \tInception Score std: %.4f\n "
      "\tFID: %.4f \tTrain CLIP Score: %.4f" % (test_is[0].item(), test_is[1].item(), test_fid.item(), test_clip))




