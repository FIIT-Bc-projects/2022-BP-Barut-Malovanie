import torch
from torchvision.transforms import v2

exp_2 = v2.Compose([
            v2.Resize(size=(128, 128), antialias=True),
            v2.CenterCrop(size=(128, 128)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


exp_2_xd = v2.Compose([
            v2.Resize(size=(64, 64), antialias=True),
            v2.CenterCrop(size=(64, 64)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

