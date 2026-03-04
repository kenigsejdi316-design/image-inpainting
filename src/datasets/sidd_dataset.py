import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.io import list_images


class SIDDDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        self.images = list_images(cfg["data_dir"])
        if split == "train":
            self.images = self.images[: int(len(self.images) * 0.9)]
        else:
            self.images = self.images[int(len(self.images) * 0.9):]

        size = cfg.get("image_size", 256)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self.noise_std_min = float(cfg.get("noise_std_min", 0.01))
        self.noise_std_max = float(cfg.get("noise_std_max", 0.06))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        clean = Image.open(self.images[idx]).convert("RGB")
        clean = self.transform(clean)
        noise = torch.randn_like(clean) * random.uniform(self.noise_std_min, self.noise_std_max)
        noisy = torch.clamp(clean + noise, 0.0, 1.0)
        return {"input": noisy, "target": clean, "task": "denoise"}
