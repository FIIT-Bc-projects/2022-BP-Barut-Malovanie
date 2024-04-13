import glob
import random
import os

from torch.utils.data import Dataset
from torchvision.io import read_image
from models import Encoder


class FlowersDataset(Dataset):

    def __init__(self, data_dir, train=True, transform=None, device="cpu"):
        self.device = device
        self.data_dir = data_dir + ("train" if train else "test") + "/"
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for class_name in os.listdir(self.data_dir):
            images = glob.glob(self.data_dir + class_name + "/*.jpg")
            for image_path in images:
                class_label = int(class_name[-4:]) - 1
                descriptions_path = image_path[: -3] + "txt"
                file = open(descriptions_path, "r")
                prompts = file.readlines()
                encoded_prompts = Encoder.encode_prompt(prompts, self.device)
                file.close()
                data.append((image_path, class_label, encoded_prompts.detach()))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path, class_label, encoded_prompts = self.data[item]
        image = read_image(image_path)
        encoded_prompt = random.choice(encoded_prompts).clone().detach()
        if self.transform:
            image = self.transform(image)
        return image, class_label, encoded_prompt


class FlowersDatasetRaw(Dataset):

    def __init__(self, data_dir, train=True, transform=None, device="cpu"):
        self.device = device
        self.data_dir = data_dir + ("train" if train else "test") + "/"
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for class_name in os.listdir(self.data_dir):
            images = glob.glob(self.data_dir + class_name + "/*.jpg")
            for image_path in images:
                class_label = int(class_name[-4:]) - 1
                descriptions_path = image_path[: -3] + "txt"
                file = open(descriptions_path, "r")
                prompts = file.readlines()
                encoded_prompts = Encoder.encode_prompt(prompts, self.device)
                file.close()
                data.append((image_path, class_label, prompts, encoded_prompts.detach()))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path, class_label, prompts, encoded = self.data[item]
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image, class_label, prompts[0], encoded[0]






