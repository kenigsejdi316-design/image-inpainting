python src/train.py --config configs/train_multitask.yaml --task low_light --epochs 1
python src/train.py --config configs/train_multitask.yaml --task denoise --epochs 1
python src/train.py --config configs/train_multitask.yaml --task super_resolution --epochs 1
python src/train.py --config configs/train_multitask.yaml --task inpainting --epochs 1 --max-train-steps 200 --max-val-steps 20
python src/train.py --config configs/train_multitask.yaml --task face_restoration --epochs 1 --max-train-steps 200 --max-val-steps 20
