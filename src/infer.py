import argparse
from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from PIL import Image
from torchvision import transforms

from src.train import build_cfg, build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_multitask.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/predictions/result.png")
    parser.add_argument("--ckpt", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_cfg(args.config)
    task_cfg = cfg["task_cfgs"][args.task]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"]).to(device)

    ckpt = args.ckpt or f"experiments/checkpoints/best_{args.task}.pth"
    ckpt_obj = torch.load(ckpt, map_location=device)
    state_dict = ckpt_obj.get("model_state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
    model.load_state_dict(state_dict)
    model.eval()

    size = task_cfg.get("image_size", 256)
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    image = Image.open(args.input).convert("RGB")
    x = tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x).squeeze(0).cpu()

    out_img = transforms.ToPILImage()(y)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
