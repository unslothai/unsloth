# Strictly local source install for Unsloth Core + Studio.
# Creates local environments and keeps installer/runtime caches inside this checkout.
[CmdletBinding()]
param(
    [string]$Python = "3.13",
    [string]$Venv = ".venv",
    [string]$Extra = "base",
    [switch]$NoTorch,
    [switch]$CoreOnly
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = if ([IO.Path]::IsPathRooted($Venv)) { $Venv } else { Join-Path $RepoRoot $Venv }

if (-not (Test-Path (Join-Path $RepoRoot "pyproject.toml"))) {
    throw "Run this from the Unsloth repository root."
}

if ($NoTorch) {
    $Extra = ""
}
if ($Extra -and $Extra -notmatch '^[a-zA-Z0-9._-]+$') {
    throw "-Extra may only contain letters, numbers, dot, underscore, and dash."
}

$env:UV_CACHE_DIR = Join-Path $RepoRoot ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $RepoRoot ".uv-python"
$env:PIP_CACHE_DIR = Join-Path $RepoRoot ".pip-cache"
$env:NPM_CONFIG_CACHE = Join-Path $RepoRoot ".npm-cache"
$env:UNSLOTH_STUDIO_HOME = Join-Path $RepoRoot ".studio"
$env:UNSLOTH_CACHE_DIR = Join-Path $env:UNSLOTH_STUDIO_HOME "cache"

$ResolvedRepo = [IO.Path]::GetFullPath($RepoRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
$ResolvedVenv = [IO.Path]::GetFullPath($VenvDir)
if (-not $ResolvedVenv.StartsWith($ResolvedRepo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "-Venv must stay inside this checkout: $RepoRoot"
}

Write-Host "Installing Unsloth from source into: $VenvDir"
Write-Host "Keeping install caches inside: $RepoRoot"

function Write-LocalActivationEnv {
    $activate = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (-not (Test-Path $activate)) { return }
    $content = Get-Content -Raw $activate
    if ($content -notmatch "Unsloth local source install") {
        Add-Content -Path $activate -Value @"

# Unsloth local source install: keep package-manager state inside this checkout.
`$env:UV_CACHE_DIR = "$($env:UV_CACHE_DIR)"
`$env:UV_PYTHON_INSTALL_DIR = "$($env:UV_PYTHON_INSTALL_DIR)"
`$env:PIP_CACHE_DIR = "$($env:PIP_CACHE_DIR)"
`$env:NPM_CONFIG_CACHE = "$($env:NPM_CONFIG_CACHE)"
`$env:UNSLOTH_STUDIO_HOME = "$($env:UNSLOTH_STUDIO_HOME)"
`$env:UNSLOTH_CACHE_DIR = "$($env:UNSLOTH_CACHE_DIR)"
"@
    }
}

$uv = Get-Command uv -ErrorAction SilentlyContinue
if ($uv) {
    & uv venv $VenvDir --python $Python --seed
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-LocalActivationEnv

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    if ($Extra) {
        & uv pip install --python $VenvPython -e "$RepoRoot[$Extra]" --torch-backend=auto
    } else {
        & uv pip install --python $VenvPython -e $RepoRoot
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "uv was not found; falling back to python venv + pip."
    Write-Host "Tip: install uv for automatic PyTorch backend selection."
    $PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
    & $PythonBin -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-LocalActivationEnv

    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($Extra) {
        & $VenvPython -m pip install -e "$RepoRoot[$Extra]"
    } else {
        & $VenvPython -m pip install -e $RepoRoot
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $VenvPython -c "import pathlib, sys; root = pathlib.Path(r'$RepoRoot').resolve(); prefix = pathlib.Path(sys.prefix).resolve(); raise SystemExit(f'ERROR: environment escaped the checkout: {prefix}') if root not in (prefix, *prefix.parents) else print(f'Python: {sys.executable}')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $CoreOnly) {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "Studio source install requires uv on PATH. Install uv first, then re-run .\install-source.ps1."
    }
    Write-Host "Installing Unsloth Studio locally into: $($env:UNSLOTH_STUDIO_HOME)"
    $InstallArgs = @("--local")
    if ($NoTorch) { $InstallArgs += "--no-torch" }
    & (Join-Path $RepoRoot "install.ps1") @InstallArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host ""
Write-Host "Done."
Write-Host "Activate with:"
Write-Host "  .\$Venv\Scripts\Activate.ps1"
Write-Host "Run Studio with:"
Write-Host "  unsloth studio -H 127.0.0.1 -p 8888"
