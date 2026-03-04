from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.io import list_images


class LOLDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        if split == "train":
            low_dir = cfg["train_low"]
            high_dir = cfg["train_high"]
        else:
            low_dir = cfg["val_low"]
            high_dir = cfg["val_high"]

        self.low_images = list_images(low_dir)
        self.high_dir = Path(high_dir)
        size = cfg.get("image_size", 256)
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_path = self.low_images[idx]
        high_path = self.high_dir / low_path.name

        low = Image.open(low_path).convert("RGB")
        high = Image.open(high_path).convert("RGB")

        low = self.transform(low)
        high = self.transform(high)
        return {"input": low, "target": high, "task": "low_light"}
