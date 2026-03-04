from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.io import list_images
from src.utils.mask import random_mask


class Places365InpaintDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        folder = cfg["train_dir"] if split == "train" else cfg["val_dir"]
        self.images = list_images(folder)
        self.size = cfg.get("image_size", 256)
        self.mask_type = cfg.get("mask_type", "mixed")
        self.mask_num_strokes = int(cfg.get("mask_num_strokes", 8))
        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        target = self.transform(image)
        mask = torch.tensor(
            random_mask(
                self.size,
                self.size,
                num_strokes=self.mask_num_strokes,
                mode=self.mask_type,
            ),
            dtype=torch.float32,
        ).unsqueeze(0)
        masked = target * (1 - mask)
        return {
            "input": masked,
            "target": target,
            "mask": mask,
            "task": "inpainting",
        }
