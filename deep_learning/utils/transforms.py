import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
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

scale_for_metrics = v2.Compose([
    v2.Resize(size=(399, 399), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

clip_img_transform = v2.Compose([
    v2.Resize(224, interpolation=InterpolationMode.BICUBIC),
    v2.CenterCrop(224),
    v2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])