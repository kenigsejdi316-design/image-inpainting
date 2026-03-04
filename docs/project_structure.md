# 项目目录说明

```text
基于深度学习的图像修复系统/
├─ data/                           # 你的原始数据集
├─ configs/
│  ├─ base.yaml                    # 训练通用配置
│  ├─ tasks.yaml                   # 数据集路径与任务映射
│  └─ train_multitask.yaml         # 多任务训练主配置
├─ src/
│  ├─ train.py                     # 训练入口
│  ├─ evaluate.py                  # 评估入口
│  ├─ infer.py                     # 推理入口
│  ├─ datasets/                    # 数据读取与预处理
│  ├─ models/                      # 网络结构
│  ├─ engine/                      # 训练与评估流程
│  ├─ metrics/                     # 指标
│  ├─ losses/                      # 损失函数
│  └─ utils/                       # 工具函数
├─ scripts/
│  ├─ train_all.ps1                # 全任务训练脚本
│  └─ eval_all.ps1                 # 全任务评估脚本
├─ experiments/
│  ├─ checkpoints/                 # 模型权重
│  ├─ logs/                        # 日志
│  └─ runs/                        # 可视化记录
├─ outputs/
│  ├─ demo/
│  └─ predictions/
├─ docs/
│  ├─ project_structure.md
│  └─ dataset_plan.md
├─ requirements.txt
└─ README.md
```
