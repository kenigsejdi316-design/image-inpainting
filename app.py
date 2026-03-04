from pathlib import Path
import sys
import uuid
import os
from typing import Any
import base64
from io import BytesIO

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from flask import Flask, render_template, request, send_from_directory
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import build_cfg, build_model
from src.utils.mask import random_mask

CONFIG_PATH = "configs/train_multitask.yaml"
TASK_LABELS = {
    "low_light": "低光增强",
    "denoise": "图像去噪",
    "super_resolution": "超分辨率",
    "inpainting": "图像补全",
    "face_restoration": "人脸修复",
    "one_click": "一键修复",
}
TASKS = list(TASK_LABELS.keys())
ONE_CLICK_TASK_ORDER = ["low_light", "denoise", "super_resolution", "inpainting", "face_restoration"]
SR_RESOLUTION_OPTIONS = {
    "same_as_input": None,
    "1280x720": (1280, 720),
    "1920x1080": (1920, 1080),
    "2560x1440": (2560, 1440),
    "3840x2160": (3840, 2160),
}
SR_RESOLUTION_LABELS = {
    "same_as_input": "与原图一致",
    "1280x720": "1280 × 720（720P）",
    "1920x1080": "1920 × 1080（1080P）",
    "2560x1440": "2560 × 1440（2K）",
    "3840x2160": "3840 × 2160（4K）",
}
UPLOAD_DIR = PROJECT_ROOT / "outputs" / "demo" / "uploads"
RESULT_DIR = PROJECT_ROOT / "outputs" / "demo" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
_MODEL_CACHE = {}


def _decode_mask_data_url(mask_data: str) -> Image.Image | None:
    if not mask_data:
        return None
    if "," not in mask_data:
        return None
    try:
        _, encoded = mask_data.split(",", 1)
        raw = base64.b64decode(encoded)
        return Image.open(BytesIO(raw)).convert("L")
    except Exception:
        return None


def _build_image_meta(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as img:
        width, height = img.size
        image_mode = img.mode
        image_format = img.format or image_path.suffix.replace(".", "").upper()
        gray = np.array(img.convert("L"), dtype=np.float32) / 255.0

    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray - smooth
    noise_sigma = float(np.std(residual) * 255.0)
    if noise_sigma < 3.0:
        noise_level = "极低"
    elif noise_sigma < 8.0:
        noise_level = "较低"
    elif noise_sigma < 15.0:
        noise_level = "中等"
    else:
        noise_level = "较高"

    file_size_kb = image_path.stat().st_size / 1024
    return {
        "width": width,
        "height": height,
        "mode": image_mode,
        "format": image_format,
        "size_kb": round(file_size_kb, 2),
        "noise_sigma": round(noise_sigma, 2),
        "noise_level": noise_level,
    }


def _auto_detect_missing_mask(image: Image.Image) -> Image.Image:
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 简单缺失区域先验：极暗/极亮 + 低纹理区域
    mean = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
    sq_mean = cv2.GaussianBlur((gray.astype(np.float32) ** 2), (0, 0), 3)
    variance = np.clip(sq_mean - mean * mean, 0, None)

    low_texture = variance < 40.0
    extreme = (gray < 22) | (gray > 245)
    cand = (low_texture & extreme).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel, iterations=2)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel, iterations=1)
    cand = cv2.dilate(cand, kernel, iterations=1)

    # 避免整图被选中，限制掩膜比例
    ratio = float(np.count_nonzero(cand)) / float(cand.size)
    if ratio > 0.35:
        cand = np.zeros_like(cand, dtype=np.uint8)

    return Image.fromarray(cand, mode="L")


def _feather_mask(mask_bin: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    if np.count_nonzero(mask_bin) == 0:
        return mask_bin.astype(np.float32)
    mask_f = mask_bin.astype(np.float32)
    soft = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    soft = np.clip(soft, 0.0, 1.0)
    return soft


def _estimate_noise_sigma_tensor(image_tensor: torch.Tensor) -> float:
    gray = image_tensor.mean(dim=0).cpu().numpy().astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = gray - smooth
    return float(np.std(residual) * 255.0)


def _apply_sr_resolution(output_img: Image.Image, input_width: int, input_height: int, sr_resolution: str) -> Image.Image:
    if sr_resolution not in SR_RESOLUTION_OPTIONS:
        sr_resolution = "same_as_input"
    target_size = SR_RESOLUTION_OPTIONS[sr_resolution]
    if target_size is None:
        target_size = (input_width, input_height)
    return output_img.resize(target_size, Image.LANCZOS)


def _run_one_click_pipeline(
    image: Image.Image,
    selected_tasks: list[str],
    ckpt_path: str | None,
    sr_resolution: str,
    auto_detect_mask: bool,
    mask_image: Image.Image | None,
    denoise_strength: float,
    low_light_strength: float,
) -> tuple[Image.Image, str, Image.Image | None]:
    current = image.convert("RGB")
    masked_preview: Image.Image | None = None
    applied_labels: list[str] = []
    skipped_labels: list[str] = []

    for sub_task in selected_tasks:
        current_w, current_h = current.size
        sub_mask = mask_image if sub_task == "inpainting" else None
        if sub_task == "inpainting" and sub_mask is None and not auto_detect_mask:
            skipped_labels.append(f"{TASK_LABELS.get(sub_task, sub_task)}（未提供掩膜）")
            continue

        out_img, _, sub_masked_preview = _predict(
            current,
            sub_task,
            ckpt_path,
            mask_image=sub_mask,
            auto_detect_mask=auto_detect_mask,
            denoise_strength=denoise_strength,
            low_light_strength=low_light_strength,
        )

        if sub_task == "super_resolution":
            out_img = _apply_sr_resolution(out_img, current_w, current_h, sr_resolution)
        else:
            if out_img.size != (current_w, current_h):
                out_img = out_img.resize((current_w, current_h), Image.BICUBIC)

        current = out_img.convert("RGB")
        applied_labels.append(TASK_LABELS.get(sub_task, sub_task))
        if sub_masked_preview is not None:
            masked_preview = sub_masked_preview

    if not applied_labels:
        raise RuntimeError("一键修复未执行任何步骤，请至少勾选一个功能；若勾选图像补全请提供掩膜或开启自动识别。")

    status = f"处理完成：一键修复（顺序：{' → '.join(applied_labels)}）"
    if "super_resolution" in selected_tasks:
        status += f"；超分输出分辨率：{SR_RESOLUTION_LABELS.get(sr_resolution, sr_resolution)}"
    if skipped_labels:
        status += f"；跳过：{'，'.join(skipped_labels)}"
    return current, status, masked_preview


def _seamless_blend(generated: Image.Image, original: Image.Image, mask_bin: np.ndarray) -> Image.Image:
    if np.count_nonzero(mask_bin) < 16:
        return generated

    src = cv2.cvtColor(np.array(generated.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)
    dst = cv2.cvtColor(np.array(original.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR)
    mask_u8 = (mask_bin * 255).astype(np.uint8)

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return generated
    center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))

    blended = cv2.seamlessClone(src, dst, mask_u8, center, cv2.NORMAL_CLONE)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    return Image.fromarray(blended_rgb)


def _load_model(task: str, ckpt_path: str | None = None):
    cfg = build_cfg(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_ckpt = Path(f"experiments/checkpoints/best_{task}.pth")
    ckpt = Path(ckpt_path).expanduser() if ckpt_path else default_ckpt
    cache_key = f"{task}:{ckpt.resolve() if ckpt.exists() else ckpt}"

    if cache_key in _MODEL_CACHE:
        model = _MODEL_CACHE[cache_key]
        return model, cfg, device, ckpt

    if not ckpt.exists():
        raise FileNotFoundError(f"未找到权重文件: {ckpt}，请先训练该任务。")

    model = build_model(cfg["model"]).to(device)
    ckpt_obj = torch.load(ckpt, map_location=device)
    state_dict = ckpt_obj.get("model_state_dict", ckpt_obj) if isinstance(ckpt_obj, dict) else ckpt_obj
    model.load_state_dict(state_dict)
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model, cfg, device, ckpt


def _predict(
    image: Image.Image,
    task: str,
    ckpt_path: str | None = None,
    mask_image: Image.Image | None = None,
    auto_detect_mask: bool = False,
    denoise_strength: float = 0.55,
    low_light_strength: float = 0.55,
) -> tuple[Image.Image, str, Image.Image | None]:
    model, cfg, device, ckpt = _load_model(task, ckpt_path)
    size = cfg["task_cfgs"][task].get("image_size", 256)
    image_rgb = image.convert("RGB")
    if task in {"inpainting", "denoise", "low_light"}:
        image_tensor = transforms.ToTensor()(image_rgb)
        _, orig_h, orig_w = image_tensor.shape
        pad_h = (4 - (orig_h % 4)) % 4
        pad_w = (4 - (orig_w % 4)) % 4
        if pad_h > 0 or pad_w > 0:
            image_tensor_model = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            image_tensor_model = image_tensor
        model_h, model_w = image_tensor_model.shape[1], image_tensor_model.shape[2]
    else:
        tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        image_tensor = tf(image_rgb)
        image_tensor_model = image_tensor
        model_h, model_w = size, size

    masked_preview = None

    if task == "inpainting":
        mask_source = "随机掩膜"
        if mask_image is not None:
            mask_arr = np.array(mask_image.convert("L").resize((model_w, model_h), Image.NEAREST))
            mask_bin = (mask_arr > 127).astype(np.float32)
            mask_source = "手动掩膜"
        elif auto_detect_mask:
            auto_mask = _auto_detect_missing_mask(image_rgb)
            mask_arr = np.array(auto_mask.resize((model_w, model_h), Image.NEAREST))
            mask_bin = (mask_arr > 127).astype(np.float32)
            if np.count_nonzero(mask_bin) == 0:
                mask_bin = random_mask(model_h, model_w).astype(np.float32)
                mask_source = "自动检测失败，已回退随机掩膜"
            else:
                mask_source = "自动识别掩膜"
        else:
            mask_bin = random_mask(model_h, model_w).astype(np.float32)

        mask_tensor = torch.from_numpy(mask_bin).unsqueeze(0)
        masked_tensor = image_tensor_model * (1.0 - mask_tensor)
        x = masked_tensor.unsqueeze(0).to(device)
    else:
        x = image_tensor_model.unsqueeze(0).to(device)
        mask_tensor = None

    with torch.no_grad():
        y = model(x).squeeze(0).cpu().clamp(0.0, 1.0)

    if task == "low_light":
        strength_level = float(np.clip(float(low_light_strength), 0.0, 1.0))
        alpha = float(np.clip(0.35 + 0.65 * strength_level, 0.35, 1.0))
        y = alpha * y + (1.0 - alpha) * image_tensor_model

        if strength_level > 0.72:
            gamma = float(np.clip(1.0 - 0.45 * (strength_level - 0.72), 0.82, 1.0))
            y = torch.clamp(torch.pow(torch.clamp(y, 1e-6, 1.0), gamma), 0.0, 1.0)

        if y.shape[1] != orig_h or y.shape[2] != orig_w:
            y = y[:, :orig_h, :orig_w]

    if task == "denoise":
        estimated_sigma = _estimate_noise_sigma_tensor(image_tensor_model)
        strength_level = float(np.clip(float(denoise_strength), 0.0, 1.0))

        extra_passes = 0
        if strength_level >= 0.7:
            extra_passes += 1
        if strength_level >= 0.9:
            extra_passes += 1
        for _ in range(extra_passes):
            with torch.no_grad():
                y_next = model(y.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0.0, 1.0)
            y = 0.82 * y_next + 0.18 * y

        alpha_min = 0.45 + 0.35 * strength_level
        alpha_max = 0.82 + 0.175 * strength_level
        alpha_base = 0.35 + 0.05 * estimated_sigma + 0.28 * strength_level
        blend_alpha = float(np.clip(alpha_base, alpha_min, alpha_max))
        y = blend_alpha * y + (1.0 - blend_alpha) * image_tensor_model

        smooth = F.avg_pool2d(image_tensor_model.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        detail = image_tensor_model - smooth
        detail_scale = 1.20 - 1.15 * strength_level
        detail_gain = float(np.clip((0.2 - 0.01 * estimated_sigma) * detail_scale, 0.0, 0.22))
        y = torch.clamp(y + detail_gain * detail, 0.0, 1.0)

        if y.shape[1] != orig_h or y.shape[2] != orig_w:
            y = y[:, :orig_h, :orig_w]

    if task == "inpainting" and mask_tensor is not None:
        sigma = float(np.clip(max(model_h, model_w) / 180.0, 2.0, 8.0))
        soft_mask_np = _feather_mask(mask_bin, sigma=sigma)
        soft_mask = torch.from_numpy(soft_mask_np).unsqueeze(0)
        y = y * soft_mask + image_tensor_model * (1.0 - soft_mask)

        if y.shape[1] != orig_h or y.shape[2] != orig_w:
            y = y[:, :orig_h, :orig_w]
            mask_bin = mask_bin[:orig_h, :orig_w]
            masked_tensor = masked_tensor[:, :orig_h, :orig_w]

        masked_preview = transforms.ToPILImage()(masked_tensor)

    out = transforms.ToPILImage()(y)
    if task == "inpainting" and mask_tensor is not None:
        out = _seamless_blend(out, transforms.ToPILImage()(image_tensor), mask_bin)

    task_label = TASK_LABELS.get(task, task)
    if task == "inpainting":
        status = (
            f"处理完成：{task_label}（{mask_source}，原图分辨率推理+软融合+无缝融合已启用），"
            f"使用权重 {ckpt}"
        )
    elif task == "denoise":
        strength_pct = int(round(float(np.clip(denoise_strength, 0.0, 1.0)) * 100.0))
        status = (
            f"处理完成：{task_label}（强度={strength_pct}%：连续可调，原图分辨率推理 + 自适应细节保护融合已启用），"
            f"使用权重 {ckpt}"
        )
    elif task == "low_light":
        strength_pct = int(round(float(np.clip(low_light_strength, 0.0, 1.0)) * 100.0))
        status = (
            f"处理完成：{task_label}（强度={strength_pct}%：连续可调，原图分辨率推理 + 自适应增强已启用），"
            f"使用权重 {ckpt}"
        )
    elif task == "face_restoration":
        status = f"处理完成：{task_label}（该任务适合已退化/模糊人脸图像，模型输入尺寸 {size}x{size}），使用权重 {ckpt}"
    else:
        status = f"处理完成：{task_label}（模型输入尺寸 {size}x{size}），使用权重 {ckpt}"
    return out, status, masked_preview


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "tasks": [{"id": task_id, "name": TASK_LABELS.get(task_id, task_id)} for task_id in TASKS],
        "one_click_task_options": [
            {"id": task_id, "name": TASK_LABELS.get(task_id, task_id)} for task_id in ONE_CLICK_TASK_ORDER
        ],
        "sr_resolution_options": [
            {"id": key, "name": SR_RESOLUTION_LABELS.get(key, key)} for key in SR_RESOLUTION_OPTIONS.keys()
        ],
        "selected_task": "low_light",
        "one_click_selected": ["low_light", "denoise", "super_resolution"],
        "sr_resolution": "same_as_input",
        "status": "",
        "input_url": None,
        "output_url": None,
        "download_url": None,
        "ckpt_path": "",
        "masked_input_url": None,
        "auto_mask_enabled": True,
        "denoise_strength": "0.55",
        "low_light_strength": "0.55",
        "input_meta": None,
        "output_meta": None,
        "compare_info": None,
    }

    if request.method == "POST":
        task = request.form.get("task", "low_light")
        if task not in TASK_LABELS:
            task = "low_light"
        one_click_selected = request.form.getlist("one_click_tasks")
        one_click_selected = [task_id for task_id in one_click_selected if task_id in ONE_CLICK_TASK_ORDER]
        if task == "one_click" and not one_click_selected:
            one_click_selected = ["low_light", "denoise", "super_resolution"]
        ckpt_path = request.form.get("ckpt_path", "").strip()
        sr_resolution = request.form.get("sr_resolution", "same_as_input").strip()
        if sr_resolution not in SR_RESOLUTION_OPTIONS:
            sr_resolution = "same_as_input"
        mask_data = request.form.get("mask_data", "").strip()
        auto_mask_enabled = request.form.get("auto_mask") == "1"
        denoise_strength_raw = request.form.get("denoise_strength", "0.55").strip().lower()
        low_light_strength_raw = request.form.get("low_light_strength", "0.55").strip().lower()
        legacy_strength_map = {"weak": 0.25, "medium": 0.55, "strong": 1.0}
        if denoise_strength_raw in legacy_strength_map:
            denoise_strength = legacy_strength_map[denoise_strength_raw]
        else:
            try:
                denoise_strength = float(denoise_strength_raw)
            except Exception:
                denoise_strength = 0.55
        denoise_strength = float(np.clip(denoise_strength, 0.0, 1.0))

        if low_light_strength_raw in legacy_strength_map:
            low_light_strength = legacy_strength_map[low_light_strength_raw]
        else:
            try:
                low_light_strength = float(low_light_strength_raw)
            except Exception:
                low_light_strength = 0.55
        low_light_strength = float(np.clip(low_light_strength, 0.0, 1.0))

        mask_file = request.files.get("mask")
        context["selected_task"] = task
        context["one_click_selected"] = one_click_selected
        context["ckpt_path"] = ckpt_path
        context["sr_resolution"] = sr_resolution
        context["auto_mask_enabled"] = auto_mask_enabled
        context["denoise_strength"] = f"{denoise_strength:.2f}"
        context["low_light_strength"] = f"{low_light_strength:.2f}"

        file = request.files.get("image")
        if file is None or file.filename == "":
            context["status"] = "请先上传图片"
            return render_template("index.html", **context)

        sample_id = uuid.uuid4().hex[:10]
        input_path = UPLOAD_DIR / f"{sample_id}_input.png"
        output_path = RESULT_DIR / f"{sample_id}_output.png"
        masked_input_path = UPLOAD_DIR / f"{sample_id}_masked_input.png"
        file.save(input_path)

        try:
            image = Image.open(input_path).convert("RGB")
            input_width, input_height = image.size
            mask_image = None
            if task == "inpainting":
                mask_image = _decode_mask_data_url(mask_data)
                if mask_image is None and mask_file is not None and mask_file.filename:
                    mask_image = Image.open(mask_file.stream).convert("L")
            elif task == "one_click" and ("inpainting" in one_click_selected):
                mask_image = _decode_mask_data_url(mask_data)
                if mask_image is None and mask_file is not None and mask_file.filename:
                    mask_image = Image.open(mask_file.stream).convert("L")

            if task == "one_click":
                output_img, status, masked_preview = _run_one_click_pipeline(
                    image,
                    one_click_selected,
                    ckpt_path if ckpt_path else None,
                    sr_resolution,
                    auto_mask_enabled,
                    mask_image,
                    denoise_strength,
                    low_light_strength,
                )
            else:
                output_img, status, masked_preview = _predict(
                    image,
                    task,
                    ckpt_path if ckpt_path else None,
                    mask_image=mask_image,
                    auto_detect_mask=auto_mask_enabled,
                    denoise_strength=denoise_strength,
                    low_light_strength=low_light_strength,
                )

            if task == "super_resolution":
                output_img = _apply_sr_resolution(output_img, input_width, input_height, sr_resolution)
                sr_label = SR_RESOLUTION_LABELS.get(sr_resolution, sr_resolution)
                status = f"{status}；输出分辨率：{sr_label}"
            elif task != "one_click":
                output_img = output_img.resize((input_width, input_height), Image.BICUBIC)
            output_img.save(output_path)

            if masked_preview is not None:
                masked_preview = masked_preview.resize((input_width, input_height), Image.NEAREST)
                masked_preview.save(masked_input_path)

            context["status"] = status
            context["input_url"] = f"/outputs/demo/uploads/{input_path.name}"
            context["output_url"] = f"/outputs/demo/results/{output_path.name}"
            if masked_preview is not None:
                context["masked_input_url"] = f"/outputs/demo/uploads/{masked_input_path.name}"
            context["download_url"] = context["output_url"]
            context["input_meta"] = _build_image_meta(input_path)
            context["output_meta"] = _build_image_meta(output_path)
            input_pixels = context["input_meta"]["width"] * context["input_meta"]["height"]
            output_pixels = context["output_meta"]["width"] * context["output_meta"]["height"]
            noise_delta = context["output_meta"]["noise_sigma"] - context["input_meta"]["noise_sigma"]
            context["compare_info"] = {
                "input_resolution": f"{context['input_meta']['width']} × {context['input_meta']['height']}",
                "output_resolution": f"{context['output_meta']['width']} × {context['output_meta']['height']}",
                "pixel_ratio": round((output_pixels / max(1, input_pixels)) * 100.0, 1),
                "noise_delta": round(noise_delta, 2),
                "noise_trend": "下降" if noise_delta < 0 else ("上升" if noise_delta > 0 else "持平"),
            }
        except Exception as exc:
            context["status"] = f"推理失败: {exc}"

    return render_template("index.html", **context)


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(PROJECT_ROOT / "outputs", filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
