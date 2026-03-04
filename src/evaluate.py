import argparse
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from src.engine.evaluator import evaluate_task
from src.models import BaselineUNet
from src.train import build_cfg, build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_multitask.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg["model"]).to(device)
    ckpt = args.ckpt or f"experiments/checkpoints/best_{args.task}.pth"
    ckpt_obj = torch.load(ckpt, map_location=device)
    state_dict = ckpt_obj.get("model_state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
    model.load_state_dict(state_dict)

    score = evaluate_task(cfg, args.task, model, device)
    print(f"Task={args.task}, PSNR={score:.3f}")


if __name__ == "__main__":
    main()
