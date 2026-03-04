from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.io import list_images


class DIV2KDataset(Dataset):
    def __init__(self, cfg: dict, split: str = "train"):
        if split == "train":
            hr_dir = cfg["train_hr"]
        else:
            hr_dir = cfg["val_hr"]

        self.hr_images = list_images(hr_dir)
        size = cfg.get("image_size", 256)
        scale = cfg.get("scale", 2)

        self.hr_tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self.lr_tf = transforms.Compose([
            transforms.Resize((size // scale, size // scale)),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_images[idx]).convert("RGB")
        target = self.hr_tf(hr)
        input_lr = self.lr_tf(hr)
        return {"input": input_lr, "target": target, "task": "super_resolution"}
