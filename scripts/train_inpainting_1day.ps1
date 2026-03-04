$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$pyCmd = "D:/anaconda/Scripts/conda.exe"
$pyArgsPrefix = @("run", "-p", "C:\Users\25766\.conda\envs\my_pyTorch", "--no-capture-output", "python")

$task = "inpainting"
# 1天内可完成的配置（通常2-6小时，依硬件）
$epochs = 24
$trainSteps = 1200
$valSteps = 120
$valEvery = 2
$numWorkers = 2
$earlyStopPatience = 6
$earlyStopMinDelta = 0.01
$saveTopK = 3
$baseArgs = @(
  "src/train.py",
  "--config", "configs/train_multitask.yaml",
  "--task", $task,
  "--epochs", "$epochs",
  "--max-train-steps", "$trainSteps",
    "--max-val-steps", "$valSteps",
    "--val-every", "$valEvery",
        "--num-workers", "$numWorkers",
        "--early-stop-patience", "$earlyStopPatience",
        "--early-stop-min-delta", "$earlyStopMinDelta",
        "--save-top-k", "$saveTopK"
)

$lastCkpt = "experiments/checkpoints/last_${task}.pth"
$bestCkpt = "experiments/checkpoints/best_${task}.pth"

if (Test-Path $lastCkpt) {
    Write-Host "[Inpainting-1Day] Resume from $lastCkpt"
    & $pyCmd @pyArgsPrefix @baseArgs --resume --ckpt $lastCkpt
}
elseif (Test-Path $bestCkpt) {
    Write-Host "[Inpainting-1Day] Resume from $bestCkpt"
    & $pyCmd @pyArgsPrefix @baseArgs --resume --ckpt $bestCkpt
}
else {
    Write-Host "[Inpainting-1Day] Train from scratch"
    & $pyCmd @pyArgsPrefix @baseArgs
}
