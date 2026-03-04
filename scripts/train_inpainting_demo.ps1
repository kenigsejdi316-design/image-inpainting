$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$task = "inpainting"
$epochs = 10
$trainSteps = 800
$valSteps = 60
$baseArgs = @("src/train.py", "--config", "configs/train_multitask.yaml", "--task", $task, "--epochs", "$epochs", "--max-train-steps", "$trainSteps", "--max-val-steps", "$valSteps")

$lastCkpt = "experiments/checkpoints/last_${task}.pth"
$bestCkpt = "experiments/checkpoints/best_${task}.pth"

if (Test-Path $lastCkpt) {
    Write-Host "[InpaintingDemo] Resume from $lastCkpt"
    python @baseArgs --resume --ckpt $lastCkpt
}
elseif (Test-Path $bestCkpt) {
    Write-Host "[InpaintingDemo] Resume from $bestCkpt"
    python @baseArgs --resume --ckpt $bestCkpt
}
else {
    Write-Host "[InpaintingDemo] Train from scratch"
    python @baseArgs
}
