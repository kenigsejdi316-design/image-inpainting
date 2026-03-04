# 实验记录（烟雾测试）

- 日期：2026-02-14
- 目的：验证多任务系统训练链路是否跑通（数据加载、前向、反向、验证、权重保存）
- 训练配置：`configs/train_multitask.yaml`

## 结果汇总

| 任务 | 数据集 | 配置 | Epoch | Loss | PSNR |
|---|---|---|---:|---:|---:|
| low_light | LOL | 默认 | 1 | 0.1660 | 14.937 |
| denoise | SIDD | 默认 | 1 | 0.2016 | 20.149 |
| super_resolution | DIV2K | 默认 | 1 | 0.1306 | 19.169 |
| inpainting | Places365 | `--max-train-steps 200 --max-val-steps 20` | 1 | 0.1048 | 22.126 |
| face_restoration | faces_dataset_small | `--max-train-steps 200 --max-val-steps 20` | 1 | 0.0705 | 26.715 |

## 说明

- Places365 数据规模非常大，完整 1 epoch 训练耗时长；烟雾测试阶段使用了步数上限以快速验证流程。
- 以上结果用于“系统可用性验证”，不代表最终收敛性能。

## 可复现实验命令

```bash
python src/train.py --config configs/train_multitask.yaml --task low_light --epochs 1
python src/train.py --config configs/train_multitask.yaml --task denoise --epochs 1
python src/train.py --config configs/train_multitask.yaml --task super_resolution --epochs 1
python src/train.py --config configs/train_multitask.yaml --task inpainting --epochs 1 --max-train-steps 200 --max-val-steps 20
python src/train.py --config configs/train_multitask.yaml --task face_restoration --epochs 1 --max-train-steps 200 --max-val-steps 20
```
