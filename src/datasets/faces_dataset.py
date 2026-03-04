import random
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.utils.io import list_images


class FacesRestorationDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        images = list_images(cfg["data_dir"])
        if split == "train":
            self.images = images[: int(len(images) * 0.9)]
        else:
            self.images = images[int(len(images) * 0.9):]

        size = cfg.get("image_size", 256)
        self.size = size
        self.cfg = cfg
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def _jpeg_compress(self, img_tensor: torch.Tensor) -> torch.Tensor:
        quality_min = self.cfg.get("jpeg_quality_min", 30)
        quality_max = self.cfg.get("jpeg_quality_max", 70)
        quality = random.randint(quality_min, quality_max)

        pil = transforms.ToPILImage()(img_tensor)
        buffer = BytesIO()
        pil.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        return transforms.ToTensor()(compressed)

    def _degrade(self, target: torch.Tensor) -> torch.Tensor:
        degraded = target.clone()

        if random.random() < self.cfg.get("downsample_prob", 0.8):
            down_scale = random.choice(self.cfg.get("downsample_scales", [2, 3, 4]))
            small_h = max(1, self.size // down_scale)
            small_w = max(1, self.size // down_scale)
            degraded = TF.resize(
                degraded,
                [small_h, small_w],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            degraded = TF.resize(
                degraded,
                [self.size, self.size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

        if random.random() < self.cfg.get("blur_prob", 0.9):
            kernel_size = random.choice(self.cfg.get("blur_kernel_sizes", [3, 5, 7]))
            sigma_min = self.cfg.get("blur_sigma_min", 0.3)
            sigma_max = self.cfg.get("blur_sigma_max", 2.0)
            sigma = random.uniform(sigma_min, sigma_max)
            blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(sigma, sigma))
            degraded = blur(degraded)

        if random.random() < self.cfg.get("color_jitter_prob", 0.6):
            degraded = TF.adjust_brightness(degraded, random.uniform(0.8, 1.2))
            degraded = TF.adjust_contrast(degraded, random.uniform(0.8, 1.2))
            degraded = TF.adjust_saturation(degraded, random.uniform(0.8, 1.2))

        if random.random() < self.cfg.get("jpeg_prob", 0.6):
            degraded = self._jpeg_compress(degraded)

        noise_std_min = self.cfg.get("noise_std_min", 0.005)
        noise_std_max = self.cfg.get("noise_std_max", 0.05)
        noise = torch.randn_like(degraded) * random.uniform(noise_std_min, noise_std_max)
        degraded = torch.clamp(degraded + noise, 0.0, 1.0)

        return degraded

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        target = self.transform(Image.open(self.images[idx]).convert("RGB"))

        degraded = self._degrade(target)

        return {"input": degraded, "target": target, "task": "face_restoration"}
