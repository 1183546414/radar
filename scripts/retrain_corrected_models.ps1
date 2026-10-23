$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "D:\conda\envs\radar\python.exe"
$logDir = Join-Path $projectRoot "outputs\logs"
$logPath = Join-Path $logDir "retrain_corrected_20261022.log"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Set-Location $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"
# 当前 RTX 3050 同时被多个进程占用，初始化模型即可能 OOM。
# 隐藏 CUDA 后由 PyTorch 稳定回退到 CPU，避免后台训练再次立即退出。
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:RADAR_AE_BATCH_SIZE = "8"
$env:RADAR_Q_BATCH_SIZE = "16"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"
$env:PYTHONUNBUFFERED = "1"

Start-Transcript -Path $logPath -Force
try {
    Write-Output "[1/3] Training corrected state encoder"
    & $pythonExe -m radar.train.state_vae_contrastive
    if ($LASTEXITCODE -ne 0) { throw "state encoder training failed: $LASTEXITCODE" }

    Write-Output "[2/3] Training latent Q model"
    & $pythonExe -m radar.train.train_q_latent
    if ($LASTEXITCODE -ne 0) { throw "latent Q training failed: $LASTEXITCODE" }

    Write-Output "[3/3] Training retrieval-conditioned Q model"
    & $pythonExe -m radar.train.train_q_retrieval
    if ($LASTEXITCODE -ne 0) { throw "retrieval Q training failed: $LASTEXITCODE" }

    Write-Output "TRAINING_COMPLETE"
}
finally {
    Stop-Transcript
}
