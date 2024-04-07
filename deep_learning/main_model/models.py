import torch
from torch import nn, no_grad
from transformers import CLIPTokenizer, CLIPTextModel


class Encoder:
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    @classmethod
    def encode_prompt(cls, prompts, device="cpu"):
        tokens = cls.tokenizer(prompts, padding=True, return_tensors="pt", truncation=True)
        cls.model.to(device)
        tokens.to(device)
        with no_grad():
            encoded = cls.model(**tokens)
        return encoded.pooler_output


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # 16x16
            nn.ConvTranspose2d(64 * 16 + 20, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, True),
            # 32x32
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, True),
            # 64x64
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, True),
            # 128x128
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 3, 4, 1, bias=False, padding="same"),
            nn.Tanh()
        )
        self.feature_maps = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 16, 8, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 16),
            nn.LeakyReLU(0.2, True),
        )
        self.prompt_features = nn.Sequential(
            nn.ConvTranspose2d(512, 20,  8, 1, 0, bias=False),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, latent, prompt):
        latent_features = self.feature_maps(latent)
        prompt_features = self.prompt_features(prompt)
        combined = torch.cat((latent_features, prompt_features), dim=1)
        return self.main(combined)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_features = nn.Sequential(
            #64x64
            nn.Conv2d(3, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8
            nn.Conv2d(64 * 8, 64 * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.prompt_features = nn.Sequential(
            nn.ConvTranspose2d(512, 20, 8, 1, 0, bias=False),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.2, True),
        )

        self.disc_out = nn.Sequential(
            nn.Conv2d(64 * 16 + 20, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.reduce_maps_classifier = nn.Sequential(
            nn.Conv2d(64 * 16 + 20, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 102),
            nn.Softmax(dim=1)
        )

    def forward(self, images, prompt):
        image_features = self.image_features(images)
        prompt_features = self.prompt_features(prompt)
        combined = torch.cat([image_features, prompt_features], dim=1)
        reduced = self.reduce_maps_classifier(combined)
        flat = torch.flatten(reduced, 1)
        classified = self.classifier(flat)
        discriminated = self.disc_out(combined)
        return discriminated, classified
