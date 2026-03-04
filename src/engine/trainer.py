import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import build_dataset
from src.metrics.image_metrics import psnr


class Trainer:
    def __init__(self, cfg: dict, task_name: str, model, optimizer, device: torch.device):
        self.cfg = cfg
        self.task_name = task_name
        self.model = model
        self.optimizer = optimizer
        self.device = device

        task_cfg = cfg["task_cfgs"][task_name]
        self.train_dataset = build_dataset(task_name, task_cfg, split="train")
        self.val_dataset = build_dataset(task_name, task_cfg, split="val")
        if len(self.train_dataset) == 0:
            raise RuntimeError(f"Train dataset is empty for task '{task_name}'. Please check paths in configs/tasks.yaml")
        if len(self.val_dataset) == 0:
            raise RuntimeError(f"Val dataset is empty for task '{task_name}'. Please check paths in configs/tasks.yaml")

        num_workers = int(cfg.get("num_workers", 4))
        pin_memory = device.type == "cuda"
        loader_common = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_common["persistent_workers"] = True
            loader_common["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=task_cfg.get("batch_size", 8),
            shuffle=True,
            **loader_common,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=task_cfg.get("batch_size", 8),
            shuffle=False,
            **loader_common,
        )

        self.criterion = torch.nn.L1Loss()
        loss_cfg = cfg.get("loss", {})
        self.l1_weight = float(loss_cfg.get("l1_weight", 1.0))
        self.ssim_weight = float(loss_cfg.get("ssim_weight", 0.1))
        self.inpaint_hole_weight = float(loss_cfg.get("inpaint_hole_weight", 6.0))
        self.inpaint_valid_weight = float(loss_cfg.get("inpaint_valid_weight", 1.0))
        self.inpaint_tv_weight = float(loss_cfg.get("inpaint_tv_weight", 0.05))

        self.use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @staticmethod
    def _ssim_index(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_pred = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
        mu_tgt = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

        sigma_pred = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_pred * mu_pred
        sigma_tgt = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_tgt * mu_tgt
        sigma_cross = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_pred * mu_tgt

        numerator = (2 * mu_pred * mu_tgt + c1) * (2 * sigma_cross + c2)
        denominator = (mu_pred * mu_pred + mu_tgt * mu_tgt + c1) * (sigma_pred + sigma_tgt + c2)
        ssim_map = numerator / (denominator + 1e-8)
        return ssim_map.mean()

    def train_one_epoch(self, max_steps: int | None = None):
        self.model.train()
        total_loss = 0.0
        step_count = 0

        for batch in tqdm(self.train_loader, desc=f"Train[{self.task_name}]"):
            inp = batch["input"].to(self.device, non_blocking=True)
            tgt = batch["target"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(inp)
                l1_loss = self.criterion(pred, tgt)
                if self.task_name == "inpainting":
                    mask = batch["mask"].to(self.device, non_blocking=True)
                    valid = 1.0 - mask

                    hole_l1 = torch.mean(torch.abs(pred - tgt) * mask)
                    valid_l1 = torch.mean(torch.abs(pred - tgt) * valid)

                    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
                    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
                    tv_loss = dx.mean() + dy.mean()

                    loss = (
                        self.inpaint_hole_weight * hole_l1
                        + self.inpaint_valid_weight * valid_l1
                        + self.inpaint_tv_weight * tv_loss
                    )
                elif self.task_name == "face_restoration":
                    ssim_score = self._ssim_index(pred, tgt)
                    loss = self.l1_weight * l1_loss + self.ssim_weight * (1.0 - ssim_score)
                else:
                    loss = l1_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            step_count += 1
            if max_steps is not None and step_count >= max_steps:
                break

        return total_loss / max(1, step_count)

    @torch.no_grad()
    def validate(self, max_steps: int | None = None):
        self.model.eval()
        total_psnr = 0.0
        step_count = 0

        for batch in tqdm(self.val_loader, desc=f"Val[{self.task_name}]"):
            inp = batch["input"].to(self.device, non_blocking=True)
            tgt = batch["target"].to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(inp)
            total_psnr += psnr(pred, tgt).item()
            step_count += 1
            if max_steps is not None and step_count >= max_steps:
                break

        return total_psnr / max(1, step_count)
