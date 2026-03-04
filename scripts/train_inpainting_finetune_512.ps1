$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$pyCmd = "D:/anaconda/Scripts/conda.exe"
$pyArgsPrefix = @("run", "-p", "C:\Users\25766\.conda\envs\my_pyTorch", "--no-capture-output", "python")

$task = "inpainting"
$epochs = 30
$trainSteps = 800
$valSteps = 80
$valEvery = 2
$numWorkers = 2
$earlyStopPatience = 4
$earlyStopMinDelta = 0.01
$saveTopK = 3

$baseArgs = @(
  "src/train.py",
  "--config", "configs/train_inpainting_512.yaml",
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

$bestCkpt = "experiments/checkpoints/best_${task}.pth"
$lastCkpt = "experiments/checkpoints/last_${task}.pth"

if (Test-Path $bestCkpt) {
    Write-Host "[Inpainting-512] Finetune from $bestCkpt"
    & $pyCmd @pyArgsPrefix @baseArgs --resume --ckpt $bestCkpt
}
elseif (Test-Path $lastCkpt) {
    Write-Host "[Inpainting-512] Finetune from $lastCkpt"
    & $pyCmd @pyArgsPrefix @baseArgs --resume --ckpt $lastCkpt
}
else {
    Write-Host "[Inpainting-512] No checkpoint found, train from scratch"
    & $pyCmd @pyArgsPrefix @baseArgs
}
