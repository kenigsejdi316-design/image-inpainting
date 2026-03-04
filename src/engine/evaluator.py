import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import build_dataset
from src.metrics.image_metrics import psnr


@torch.no_grad()
def evaluate_task(cfg: dict, task_name: str, model, device: torch.device):
    task_cfg = cfg["task_cfgs"][task_name]
    dataset = build_dataset(task_name, task_cfg, split="val")
    loader = DataLoader(
        dataset,
        batch_size=task_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    model.eval()
    total_psnr = 0.0
    for batch in tqdm(loader, desc=f"Eval[{task_name}]"):
        inp = batch["input"].to(device)
        tgt = batch["target"].to(device)
        pred = model(inp)
        total_psnr += psnr(pred, tgt).item()

    return total_psnr / max(1, len(loader))
