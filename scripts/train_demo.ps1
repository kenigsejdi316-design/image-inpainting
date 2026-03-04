$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

function Start-TaskTraining($task, $epochs, $trainSteps, $valSteps) {
    $baseArgs = @("src/train.py", "--config", "configs/train_multitask.yaml", "--task", $task, "--epochs", "$epochs", "--max-train-steps", "$trainSteps", "--max-val-steps", "$valSteps")

    $lastCkpt = "experiments/checkpoints/last_${task}.pth"
    $bestCkpt = "experiments/checkpoints/best_${task}.pth"

    if (Test-Path $lastCkpt) {
        Write-Host "[DemoTrain] Resume from $lastCkpt"
        python @baseArgs --resume --ckpt $lastCkpt
    }
    elseif (Test-Path $bestCkpt) {
        Write-Host "[DemoTrain] Resume from $bestCkpt"
        python @baseArgs --resume --ckpt $bestCkpt
    }
    else {
        Write-Host "[DemoTrain] Train from scratch: $task"
        python @baseArgs
    }
}

# 低耗时可演示配置（保证前端可稳定出图）
Start-TaskTraining "low_light" 8 300 30
Start-TaskTraining "denoise" 8 300 30
Start-TaskTraining "super_resolution" 8 400 40
Start-TaskTraining "inpainting" 8 600 50
Start-TaskTraining "face_restoration" 8 300 30

Write-Host "[DemoTrain] All demo tasks completed."
