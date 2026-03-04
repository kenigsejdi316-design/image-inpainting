import argparse
from pathlib import Path
import sys
import os
import csv
import logging
from datetime import datetime
import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.engine.trainer import Trainer
from src.models import BaselineUNet


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_cfg(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    base_cfg = load_yaml(cfg["defaults"]["base"])
    tasks_cfg = load_yaml(cfg["defaults"]["tasks"])

    merged = {}
    merged.update(base_cfg)
    merged.update(cfg.get("training", {}))
    merged["model"] = cfg.get("model", {})
    merged["loss"] = cfg.get("loss", {})
    merged["task_cfgs"] = tasks_cfg.get("tasks", {})
    merged["enabled_tasks"] = cfg.get("tasks", {}).get("enabled", [])
    return merged


def build_model(model_cfg: dict) -> BaselineUNet:
    model_args = {k: v for k, v in model_cfg.items() if k != "name"}
    return BaselineUNet(**model_args)


def _extract_model_state(ckpt_obj: dict) -> dict:
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


def load_checkpoint(ckpt_path: Path, model, optimizer=None, device: torch.device | None = None) -> dict:
    map_location = device if device is not None else "cpu"
    ckpt_obj = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(_extract_model_state(ckpt_obj))

    loaded_epoch = 0
    best_psnr = -1.0
    if isinstance(ckpt_obj, dict):
        loaded_epoch = int(ckpt_obj.get("epoch", 0) or 0)
        best_psnr = float(ckpt_obj.get("best_psnr", -1.0))
        if optimizer is not None and "optimizer_state_dict" in ckpt_obj:
            optimizer.load_state_dict(ckpt_obj["optimizer_state_dict"])

    return {"epoch": loaded_epoch, "best_psnr": best_psnr}


def save_checkpoint(ckpt_path: Path, task_name: str, epoch: int, best_psnr: float, model, optimizer):
    ckpt_obj = {
        "task": task_name,
        "epoch": epoch,
        "best_psnr": float(best_psnr),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(ckpt_obj, ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_multitask.yaml")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--save-top-k", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    return parser.parse_args()


def build_task_logger(task_name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"train.{task_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / f"train_{task_name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def append_metrics_csv(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "task", "epoch", "train_loss", "val_psnr", "lr"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    cfg = build_cfg(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.no_amp:
        cfg["use_amp"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get("lr", 2e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    enabled_tasks = [args.task] if args.task else cfg["enabled_tasks"]

    save_dir = Path(cfg.get("save_dir", "experiments/checkpoints"))
    log_dir = Path(cfg.get("log_dir", "experiments/logs"))
    run_dir = Path(cfg.get("run_dir", "experiments/runs"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    for task_name in enabled_tasks:
        trainer = Trainer(cfg, task_name, model, optimizer, device)
        logger = build_task_logger(task_name, log_dir)
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_writer = SummaryWriter(log_dir=str(run_dir / f"tb_{task_name}_{time_tag}"))
        csv_path = run_dir / f"metrics_{task_name}.csv"

        print(f"\n===== Task: {task_name} =====")
        logger.info(f"Start task={task_name}, device={device}")

        best_psnr = -1.0
        start_epoch = 1
        if args.resume:
            resume_ckpt = Path(args.ckpt).expanduser() if args.ckpt else (save_dir / f"last_{task_name}.pth")
            if not resume_ckpt.exists():
                raise FileNotFoundError(f"未找到断点权重: {resume_ckpt}")
            resume_info = load_checkpoint(resume_ckpt, model, optimizer, device)
            start_epoch = resume_info["epoch"] + 1 if resume_info["epoch"] > 0 else 1
            best_psnr = resume_info["best_psnr"]
            logger.info(
                f"resume from checkpoint={resume_ckpt}, loaded_epoch={resume_info['epoch']}, "
                f"start_epoch={start_epoch}, best_psnr={best_psnr:.6f}"
            )
            print(
                f"Resume: task={task_name}, ckpt={resume_ckpt}, "
                f"loaded_epoch={resume_info['epoch']}, start_epoch={start_epoch}"
            )

        total_epochs = cfg.get("epochs", 50)
        val_every = max(1, int(args.val_every if args.val_every is not None else cfg.get("val_every", 1)))
        early_stop_patience = int(
            args.early_stop_patience
            if args.early_stop_patience is not None
            else cfg.get("early_stop_patience", 0)
        )
        early_stop_min_delta = float(
            args.early_stop_min_delta
            if args.early_stop_min_delta is not None
            else cfg.get("early_stop_min_delta", 0.0)
        )
        save_top_k = max(
            0,
            int(args.save_top_k if args.save_top_k is not None else cfg.get("save_top_k", 1)),
        )
        no_improve_count = 0
        top_k_ckpts: list[tuple[float, Path]] = []

        for epoch in range(start_epoch, total_epochs + 1):
            train_loss = trainer.train_one_epoch(max_steps=args.max_train_steps)
            should_validate = (epoch % val_every == 0) or (epoch == total_epochs)
            val_psnr = trainer.validate(max_steps=args.max_val_steps) if should_validate else float("nan")

            if should_validate:
                print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | psnr={val_psnr:.3f}")
            else:
                print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | psnr=skip")

            lr = optimizer.param_groups[0]["lr"]
            tb_writer.add_scalar("train/loss", train_loss, epoch)
            if should_validate:
                tb_writer.add_scalar("val/psnr", val_psnr, epoch)
            tb_writer.add_scalar("train/lr", lr, epoch)

            append_metrics_csv(
                csv_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "task": task_name,
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "val_psnr": "" if math.isnan(val_psnr) else f"{val_psnr:.6f}",
                    "lr": f"{lr:.8f}",
                },
            )
            if should_validate:
                logger.info(
                    f"epoch={epoch:03d}, loss={train_loss:.6f}, psnr={val_psnr:.6f}, lr={lr:.8f}"
                )
            else:
                logger.info(
                    f"epoch={epoch:03d}, loss={train_loss:.6f}, psnr=skip, lr={lr:.8f}"
                )

            last_ckpt_path = save_dir / f"last_{task_name}.pth"
            save_checkpoint(last_ckpt_path, task_name, epoch, best_psnr, model, optimizer)

            if should_validate:
                if save_top_k > 0:
                    worst_score = top_k_ckpts[-1][0] if len(top_k_ckpts) >= save_top_k else None
                    should_save_top = (worst_score is None) or (val_psnr > worst_score)
                    if should_save_top:
                        top_path = save_dir / f"top_{task_name}_{time_tag}_epoch{epoch:03d}_psnr{val_psnr:.4f}.pth"
                        save_checkpoint(top_path, task_name, epoch, best_psnr, model, optimizer)
                        top_k_ckpts.append((val_psnr, top_path))
                        top_k_ckpts.sort(key=lambda item: item[0], reverse=True)
                        while len(top_k_ckpts) > save_top_k:
                            _, rm_path = top_k_ckpts.pop()
                            if rm_path.exists():
                                rm_path.unlink()

                if val_psnr > (best_psnr + early_stop_min_delta):
                    best_psnr = val_psnr
                    ckpt_path = save_dir / f"best_{task_name}.pth"
                    save_checkpoint(ckpt_path, task_name, epoch, best_psnr, model, optimizer)
                    logger.info(f"saved best checkpoint: {ckpt_path}")
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if early_stop_patience > 0 and no_improve_count >= early_stop_patience:
                    logger.info(
                        f"early stopped at epoch={epoch:03d}, patience={early_stop_patience}, "
                        f"min_delta={early_stop_min_delta:.6f}"
                    )
                    print(
                        f"Early Stop: task={task_name}, epoch={epoch:03d}, "
                        f"patience={early_stop_patience}"
                    )
                    break

        tb_writer.flush()
        tb_writer.close()
        logger.info(f"Finish task={task_name}, best_psnr={best_psnr:.6f}")


if __name__ == "__main__":
    main()
