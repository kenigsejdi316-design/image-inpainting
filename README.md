# 基于深度学习的图像修复系统（毕业设计）

本项目面向 **图像修复统一框架**，结合你已收集的 5 类数据集，支持以下任务：
并附带数据集链接
- 低照度增强（LOL）https://www.kaggle.com/datasets/soumikrakshit/lol-dataset
- 图像去噪（SIDD）https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset
- 图像超分辨率（DIV2K）https://www.kaggle.com/datasets/joe1995/div2k-dataset?resource=download
- 场景图像修复/补全（Places365 + 随机掩膜）https://www.kaggle.com/datasets/benjaminkz/places365?select=train.txt
- 人脸图像修复（faces_dataset_small）https://www.kaggle.com/datasets/tommykamaz/faces-dataset-small

## 1. 项目结构

详见 [docs/project_structure.md](docs/project_structure.md)。

## 2. 数据集映射

详见 [docs/dataset_plan.md](docs/dataset_plan.md)。

实验烟雾测试记录见 [docs/experiment_log.md](docs/experiment_log.md)。

## 3. 快速开始

```bash
pip install -r requirements.txt
python src/train.py --config configs/train_multitask.yaml
```

单任务训练（示例）：

```bash
python src/train.py --config configs/train_multitask.yaml --task low_light
python src/train.py --config configs/train_multitask.yaml --task denoise
```

人脸修复增强训练（已启用更真实退化 + L1+SSIM 组合损失）：

```bash
python src/train.py --config configs/train_multitask.yaml --task face_restoration --epochs 50 --resume --ckpt experiments/checkpoints/last_face_restoration.pth
```

训练过程会自动记录到：

- `experiments/logs/train_<task>.log`（文本日志）
- `experiments/runs/metrics_<task>.csv`（每轮指标）
- `experiments/runs/tb_<task>_时间戳/`（TensorBoard 可视化）

查看可视化曲线：

```bash
tensorboard --logdir experiments/runs
```

浏览器打开 `http://localhost:6006`

快速烟雾测试（推荐先跑通流程）：

```bash
python src/train.py --config configs/train_multitask.yaml --task inpainting --epochs 1 --max-train-steps 200 --max-val-steps 20
```

可演示版训练（推荐，耗时大幅缩短且前端可稳定出图）：

```bash
powershell -ExecutionPolicy Bypass -File scripts/train_demo.ps1
```

仅快速优化图像补全：

```bash
powershell -ExecutionPolicy Bypass -File scripts/train_inpainting_demo.ps1
```

图像补全优化重训（1天内可完成，推荐）：

```bash
powershell -ExecutionPolicy Bypass -File scripts/train_inpainting_1day.ps1
```

说明：该脚本已启用“掩膜区域加权损失 + 边界平滑损失”，通常在 2~6 小时完成（与硬件相关）。

可选极速参数（进一步提速）：

```bash
python src/train.py --config configs/train_multitask.yaml --task inpainting --epochs 24 --max-train-steps 1200 --max-val-steps 120 --val-every 2 --num-workers 2 --early-stop-patience 6 --early-stop-min-delta 0.01 --save-top-k 3 --resume --ckpt experiments/checkpoints/last_inpainting.pth
```

512 分辨率微调（进一步提升补全细节，推荐在 1day 训练后执行）：

```bash
powershell -ExecutionPolicy Bypass -File scripts/train_inpainting_finetune_512.ps1
```

也可一键执行：

```bash
powershell -ExecutionPolicy Bypass -File scripts/smoke_test.ps1
```

评估与推理：

```bash
python src/evaluate.py --config configs/train_multitask.yaml --task low_light
python src/infer.py --config configs/train_multitask.yaml --task inpainting --input path/to/image.jpg
```

## 5. 前端界面展示（Flask）

先确保至少训练过一个任务并生成权重，例如：

```bash
python src/train.py --config configs/train_multitask.yaml --task low_light --epochs 1
```

启动前端：

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:7860`，即可上传图片并选择任务进行演示。

前端详细说明见 [docs/frontend_usage.md](docs/frontend_usage.md)。

若遇到 OpenMP 冲突（`libiomp5md.dll already initialized`），请使用：

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_app.ps1
```

也可一键执行：

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_app.ps1
```

## 4. 你的数据目录要求

当前默认数据根目录为 `data/`，并假设以下路径存在：

- `data/lol_dataset/our485/low`, `data/lol_dataset/our485/high`
- `data/SIDD_Small_sRGB_Only/Data`
- `data/DIV2K/DIV2K_train_HR`, `data/DIV2K/DIV2K_valid_HR`
- `data/place365/train`
- `data/faces_dataset_small`

如果路径不同，修改 `configs/tasks.yaml` 即可。
