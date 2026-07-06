# docker_confirm.ps1  (Unsloth Docker image confirmation - Windows)
# Confirms the published Unsloth Docker images actually work on this machine
# through Docker Desktop: pulls them, checks WSL2 GPU passthrough (or CPU
# fallback), runs a real 5-step LoRA training smoke, checks the baked
# llama.cpp GGUF tooling, boots the full image and probes Studio +
# JupyterLab, then prints a PASS/FAIL report.
#
# One-liner (PowerShell):
#   irm https://raw.githubusercontent.com/unslothai/unsloth/main/docker/docker_confirm.ps1 | iex
#
# What to expect per machine class:
#   Windows + NVIDIA (RTX 5070 / DGX Spark): GPU mode when Docker Desktop
#     uses the WSL2 backend with GPU support enabled (Settings > Resources).
#   Windows + AMD (Strix Halo): CPU mode - Docker Desktop has no ROCm
#     passthrough; training phases are skipped, Studio chat / Jupyter / GGUF
#     tooling still validate. Use the native install for AMD GPU work.
#
# Env overrides: $env:IMAGE, $env:BASE_IMAGE, $env:GPUS ('auto'|'all'|'none'),
#   $env:PORT_STUDIO (18000), $env:PORT_JUPYTER (18888), $env:WORK,
#   $env:SKIP_PULL, $env:SKIP_TRAIN, $env:KEEP

$ErrorActionPreference = "Continue"
$IMAGE        = if ($env:IMAGE)        { $env:IMAGE }        else { "unsloth/unsloth:latest" }
$BASE_IMAGE   = if ($env:BASE_IMAGE)   { $env:BASE_IMAGE }   else { "unsloth/unsloth:core" }
$GPUS         = if ($env:GPUS)         { $env:GPUS }         else { "auto" }
$PORT_STUDIO  = if ($env:PORT_STUDIO)  { $env:PORT_STUDIO }  else { 18000 }
$PORT_JUPYTER = if ($env:PORT_JUPYTER) { $env:PORT_JUPYTER } else { 18888 }
$WORK         = if ($env:WORK)         { $env:WORK }         else { Join-Path $HOME "unsloth_docker_test" }
$SKIP_PULL    = $env:SKIP_PULL -eq "1"
$SKIP_TRAIN   = $env:SKIP_TRAIN -eq "1"
$KEEP         = $env:KEEP -eq "1"

$script:PASS_N = 0; $script:FAIL_N = 0; $script:WARN_N = 0; $script:STUDIO_CID = ""
function Bold($m){ Write-Host $m -ForegroundColor White }
function Ok($m)  { Write-Host "  [PASS] $m" -ForegroundColor Green;  $script:PASS_N++ }
function Bad($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red;    $script:FAIL_N++ }
function Warn($m){ Write-Host "  [WARN] $m" -ForegroundColor Yellow; $script:WARN_N++ }
function Info($m){ Write-Host "         $m" }
function Hr()    { Write-Host ("-" * 63) }

New-Item -ItemType Directory -Force -Path $WORK | Out-Null
Write-Host ""; Bold "=== Unsloth Docker image confirmation (Windows) ==="
Write-Host "scratch dir : $WORK"; Hr

# 1) Host detection -----------------------------------------------------------
Bold "1) Host detection"
Info ("windows   : " + [System.Environment]::OSVersion.VersionString + " " + $env:PROCESSOR_ARCHITECTURE)
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  Bad "docker not found - install Docker Desktop first"
  Bold "RESULT: cannot continue without docker."; exit 1
}
docker info *> $null
if ($LASTEXITCODE -ne 0) {
  Bad "docker daemon not reachable - start Docker Desktop"
  Bold "RESULT: cannot continue without a reachable docker daemon."; exit 1
}
Ok ("docker daemon reachable (" + (docker --version) + ")")
$osType = (docker info --format "{{.OSType}}" 2>$null)
if ($osType -ne "linux") {
  Bad "Docker Desktop is in Windows-container mode (OSType=$osType) - switch to Linux containers"
}

$GPU_MODE = $false
if ($GPUS -eq "none") {
  Info "GPU mode  : disabled by GPUS=none"
} elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $gpus = nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>$null
  if ($LASTEXITCODE -eq 0 -and $gpus) {
    $gpus | ForEach-Object { Info ("           - " + $_) }
    Ok "NVIDIA GPU visible on the host - probing WSL2 passthrough below"
    $GPU_MODE = $true
  } else {
    Info "nvidia-smi present but no GPU listed"
  }
} else {
  Info "no NVIDIA GPU on the host (or nvidia-smi missing)"
}
if (-not $GPU_MODE) {
  Warn "CPU mode: training phases are skipped; Studio chat / Jupyter / GGUF tooling still validate"
}
Hr

# 2) Pull images --------------------------------------------------------------
Bold "2) Pull images"
foreach ($img in @($BASE_IMAGE, $IMAGE)) {
  if ($SKIP_PULL) {
    docker image inspect $img *> $null
    if ($LASTEXITCODE -eq 0) { Ok "local image present: $img" } else { Bad "SKIP_PULL=1 but image missing locally: $img" }
  } else {
    $log = Join-Path $WORK ("pull_" + ($img -replace "[/:]", "_") + ".log")
    docker pull $img *> $log
    if ($LASTEXITCODE -eq 0) { Ok "pulled $img" }
    else {
      docker image inspect $img *> $null
      # Locally built tags are not on a registry; presence is what matters.
      if ($LASTEXITCODE -eq 0) { Warn "not pullable but present locally: $img" }
      else { Bad "could not pull $img (see $log)" }
    }
  }
}
Hr

# 3) Container runtime check --------------------------------------------------
Bold "3) Container runtime check"
# Mirror docker_confirm.sh's GPU selector translation: bare indices and
# comma lists become device= selectors (Docker reads a bare integer for
# --gpus as a COUNT, not an index). Built as an args array so every docker
# run call splats it identically.
#
# Comma lists are special: docker CSV-parses the --gpus value, so a list
# must arrive as a literal "device=0,1" INCLUDING the double quotes. How
# PowerShell passes embedded quotes to native commands changed in 7.3
# (PSNativeCommandArgumentPassing), so pick the escaping per version;
# single selectors need no quoting anywhere.
$GPU_SELECTOR = "all"
if ($GPUS -notin @("auto", "all", "none")) {
  $sel = $GPUS -replace "^device=", ""
  if ($sel -match ",") {
    if ($PSVersionTable.PSVersion -ge [version]"7.3") { $GPU_SELECTOR = '"device=' + $sel + '"' }
    else                                              { $GPU_SELECTOR = '\"device=' + $sel + '\"' }
  } else {
    $GPU_SELECTOR = "device=$sel"
  }
}
$GpuRunArgs = @("--gpus", $GPU_SELECTOR)
if ($GPU_MODE) {
  $log = Join-Path $WORK "gpu_check.log"
  docker run --rm @GpuRunArgs $BASE_IMAGE python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, '-', torch.cuda.get_device_name(0))" *> $log
  if ($LASTEXITCODE -eq 0) {
    Ok ("torch.cuda available in-container: " + (Get-Content $log -Tail 1))
  } else {
    Bad "GPU passthrough failed (see $log) - check Docker Desktop WSL2 GPU support; falling back to CPU mode"
    Get-Content $log -Tail 5 | ForEach-Object { Info $_ }
    $GPU_MODE = $false
  }
}
if (-not $GPU_MODE) {
  $log = Join-Path $WORK "cpu_check.log"
  docker run --rm -e UNSLOTH_ALLOW_CPU=1 $BASE_IMAGE python -c "import torch; print('torch', torch.__version__, 'cpu-mode ok')" *> $log
  if ($LASTEXITCODE -eq 0) {
    Ok ("CPU mode boots: " + (Get-Content $log -Tail 1))
  } else {
    Bad "container failed to start even in CPU mode (see $log)"
    Get-Content $log -Tail 5 | ForEach-Object { Info $_ }
  }
}
Hr

# 4) Training smoke (GPU only) ------------------------------------------------
Bold "4) Training smoke"
if ($GPU_MODE -and -not $SKIP_TRAIN) {
  $log = Join-Path $WORK "train_smoke.log"
  $hfArgs = @(); if ($env:HF_TOKEN) { $hfArgs = @("-e", "HF_TOKEN") }
  docker run --rm @GpuRunArgs --ipc=host @hfArgs $BASE_IMAGE python /workspace/smoke_test.py *> $log
  if ($LASTEXITCODE -eq 0) {
    Ok "smoke_test.py: 5 LoRA steps completed"
    Select-String -Path $log -Pattern "^step|loss" | Select-Object -Last 5 | ForEach-Object { Info $_.Line }
  } else {
    Bad "training smoke failed (see $log)"
    Get-Content $log -Tail 10 | ForEach-Object { Info $_ }
  }
} else {
  Warn "skipped (CPU mode or SKIP_TRAIN=1)"
}
Hr

# 5) GGUF tooling -------------------------------------------------------------
Bold "5) GGUF tooling (baked llama.cpp)"
$log = Join-Path $WORK "gguf_check.log"
docker run --rm -e UNSLOTH_SKIP_GPU_CHECK=1 $BASE_IMAGE bash -c 'set -e; test -x "$UNSLOTH_LLAMA_CPP_PATH/llama-quantize"; test -f "$UNSLOTH_LLAMA_CPP_PATH/convert_hf_to_gguf.py"; "$UNSLOTH_LLAMA_CPP_PATH/llama-server" --version 2>&1 | head -2' *> $log
if ($LASTEXITCODE -eq 0) {
  Ok "llama-quantize + llama-server + convert_hf_to_gguf.py present and runnable"
  Select-String -Path $log -Pattern "version" | Select-Object -First 2 | ForEach-Object { Info $_.Line }
} else {
  Bad "baked llama.cpp check failed (see $log)"
  Get-Content $log -Tail 5 | ForEach-Object { Info $_ }
}
Hr

# 5b) vLLM (GRPO fast_inference=True) -----------------------------------------
Bold "5b) vLLM (GRPO fast_inference=True)"
$log = Join-Path $WORK "vllm_check.log"
docker run --rm -e UNSLOTH_SKIP_GPU_CHECK=1 $BASE_IMAGE python -c 'import vllm; print("vllm", vllm.__version__)' *> $log
if ($LASTEXITCODE -eq 0) {
  Ok ("vllm importable: " + (Get-Content $log -Tail 1))
} else {
  $imgArch = docker run --rm -e UNSLOTH_SKIP_GPU_CHECK=1 $BASE_IMAGE uname -m 2>$null
  if ($imgArch -eq "x86_64") {
    Bad "vllm missing or broken on x86_64 image (see $log)"
    Get-Content $log -Tail 3 | ForEach-Object { Info $_ }
  } else {
    Warn "vllm not available on $imgArch image; GRPO fast_inference=True unavailable (arm64 wheels are newer, fail-soft at image build)"
  }
}
Hr

# 6) Studio + JupyterLab ------------------------------------------------------
Bold "6) Studio + JupyterLab (full image)"
$runArgs = @("-d", "-p", "${PORT_STUDIO}:8000", "-p", "${PORT_JUPYTER}:8888")
if ($GPU_MODE) { $runArgs += $GpuRunArgs } else { $runArgs += @("-e", "UNSLOTH_ALLOW_CPU=1") }
$script:STUDIO_CID = (docker run @runArgs $IMAGE 2>(Join-Path $WORK "studio_run.err"))
if (-not $script:STUDIO_CID) {
  Bad ("full image failed to start (see " + (Join-Path $WORK "studio_run.err") + ")")
} else {
  Info ("container : " + $script:STUDIO_CID.Substring(0, 12) + " (studio http://localhost:$PORT_STUDIO, jupyter http://localhost:$PORT_JUPYTER)")
  $okStudio = $false; $okJupyter = $false
  foreach ($i in 1..60) {
    if (-not $okStudio)  { try { Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$PORT_STUDIO/api/health" -TimeoutSec 4 | Out-Null; $okStudio = $true } catch {} }
    # /login, not /api: a password hash is always configured so /api returns 403.
    if (-not $okJupyter) { try { Invoke-WebRequest -UseBasicParsing -Uri "http://localhost:$PORT_JUPYTER/login"     -TimeoutSec 4 | Out-Null; $okJupyter = $true } catch {} }
    if ($okStudio -and $okJupyter) { break }
    Start-Sleep -Seconds 5
  }
  if ($okStudio)  { Ok "Studio /api/health healthy" } else { Bad "Studio /api/health never went healthy (docker logs $($script:STUDIO_CID.Substring(0,12)))"; docker logs --tail 15 $script:STUDIO_CID 2>&1 | ForEach-Object { Info $_ } }
  if ($okJupyter) { Ok "JupyterLab /login responding" } else { Bad "JupyterLab /login never responded" }
}
Hr

# Summary ---------------------------------------------------------------------
Bold "=== SUMMARY ==="
Write-Host "images    : $IMAGE / $BASE_IMAGE"
Write-Host ("gpu_mode  : " + $GPU_MODE)
Write-Host "logs      : $WORK"
Write-Host "PASS: $script:PASS_N   WARN: $script:WARN_N   FAIL: $script:FAIL_N"
if (-not $KEEP -and $script:STUDIO_CID) { docker rm -f $script:STUDIO_CID *> $null }
elseif ($KEEP -and $script:STUDIO_CID)  { Write-Host ("container " + $script:STUDIO_CID.Substring(0,12) + " left running (KEEP=1): studio :$PORT_STUDIO jupyter :$PORT_JUPYTER") }
if ($script:FAIL_N -eq 0) {
  Bold "RESULT: CONFIRMED - the Unsloth Docker images work on this machine."
  exit 0
} else {
  Bold "RESULT: $script:FAIL_N hard failure(s) - paste this whole output back."
  exit 1
}
