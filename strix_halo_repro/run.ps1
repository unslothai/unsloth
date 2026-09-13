<#
    Strix Halo memory reporting: what every layer says this GPU has.

    Run this on a Windows Ryzen AI Max (gfx1150/gfx1151) box, ideally with
    Variable Graphics Memory set high (96 GiB is the configuration AMD reported
    against). Everything runs from the current user: no admin, no pip install,
    no Docker. The GPU is touched only by the allocation probe and, if you pass
    -ModelPath, by one model load.

    Default run downloads two llama.cpp release zips (~200 MB) and takes a few
    minutes. -ModelPath additionally loads a model you already have.

        powershell -ExecutionPolicy Bypass -File .\run.ps1
        powershell -ExecutionPolicy Bypass -File .\run.ps1 -ModelPath D:\gguf\model-00001-of-00004.gguf

    Send back the whole output directory: it is printed at the end.
#>
[CmdletBinding()]
param(
    [string] $Tag              = "b10798-mix-659e406",
    [string] $Work             = "$env:USERPROFILE\strix_halo_repro",
    [string] $ModelPath        = "",
    [string] $PatchedDllUrl    = "",
    [string] $PatchedDllSha256 = "",
    [string] $Python           = ""
)

$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$rel  = "https://github.com/unslothai/llama.cpp/releases/download"
$out  = Join-Path $Work "out"
New-Item -ItemType Directory -Force -Path $out | Out-Null

function Fail($msg) {
    Write-Host ""
    Write-Host "FAILED: $msg" -ForegroundColor Red
    Write-Host "Nothing further ran. Send $out anyway; a failure this early is itself the finding."
    exit 1
}

# --- Python -----------------------------------------------------------------
# The probes are stdlib only, so any 3.9+ interpreter does. Named explicitly
# rather than trusting PATH: a Microsoft Store stub on PATH answers `python`
# and then refuses to run, which reads as a broken probe rather than a missing
# interpreter.
if ($Python -and (Test-Path $Python)) {
    $py = $Python
} else {
    $py = $null
    foreach ($c in @("C:\Program Files\Python313\python.exe",
                     "C:\Program Files\Python312\python.exe",
                     "C:\Program Files\Python311\python.exe",
                     "C:\Program Files\Python310\python.exe")) {
        if (Test-Path $c) { $py = $c; break }
    }
    if (-not $py) {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd -and $cmd.Source -notlike "*WindowsApps*") { $py = $cmd.Source }
    }
}
if (-not $py) { Fail "no Python 3 found. Install from python.org, or pass -Python C:\path\to\python.exe" }
& $py -c "import sys; assert sys.version_info >= (3, 9)" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Fail "the Python at $py is older than 3.9" }
Write-Host "python: $py"

# --- 1. what the machine is -------------------------------------------------
Write-Host ""
Write-Host "== 1. the host =============================================================="
& $py "$here\probe.py" --state host --sections host --out "$out\host.json"
if ($LASTEXITCODE -ne 0) { Fail "the host probe did not run" }
$h = Get-Content "$out\host.json" -Encoding UTF8 | ConvertFrom-Json
$ramB = [double]$h.sections.host.total_phys_bytes
$vgmB = 0
foreach ($ad in $h.sections.host.adapters) {
    if ($ad.qw_memory_size -and ([double]$ad.qw_memory_size) -gt $vgmB) {
        $vgmB = [double]$ad.qw_memory_size
        Write-Host ("GPU: {0}  driver {1}" -f $ad.name, $ad.driver_version)
    }
}
if ($vgmB -le 0) {
    # Not fatal: every later cell reads its own numbers, and a machine whose
    # carve-out cannot be read is itself worth reporting rather than refusing.
    Write-Host "WARNING: no graphics carve-out found in the registry." -ForegroundColor Yellow
    Write-Host "The over-report statement needs it, so that statement will be undecided." -ForegroundColor Yellow
}
$ram = [math]::Round($ramB / 1GB, 2)
$vgm = [math]::Round($vgmB / 1GB, 2)
$machine = [math]::Round(($ramB + $vgmB) / 1GB, 2)
Write-Host "carve-out (registry)  : $vgm GiB"
Write-Host "RAM Windows can see   : $ram GiB"
Write-Host "so the machine holds  : $machine GiB"
if ($vgm -lt 80) {
    Write-Host ""
    Write-Host "NOTE: the carve-out is $vgm GiB. The over-report AMD described needs a LARGE" -ForegroundColor Yellow
    Write-Host "carve-out (96 GiB was theirs). This run is still useful, but if the numbers" -ForegroundColor Yellow
    Write-Host "come back consistent that may only mean the carve-out is too small to show it." -ForegroundColor Yellow
}

# --- 2. the llama.cpp builds ------------------------------------------------
Write-Host ""
Write-Host "== 2. llama.cpp $Tag ========================================================"
& $py "$here\fetch_llamacpp.py" --url "$rel/$Tag/app-$Tag-windows-x64-rocm-gfx1151.zip" --dest "$Work\lcpp\rocm"   --out "$out\bin_rocm.json"
if ($LASTEXITCODE -ne 0) { Fail "the ROCm build did not download" }
& $py "$here\fetch_llamacpp.py" --url "$rel/$Tag/app-$Tag-windows-x64-vulkan.zip"        --dest "$Work\lcpp\vulkan" --out "$out\bin_vulkan.json"
if ($LASTEXITCODE -ne 0) { Fail "the Vulkan build did not download" }
$rocmBin   = (Get-Content "$out\bin_rocm.json"   -Encoding UTF8 | ConvertFrom-Json).bin_dir
$vulkanBin = (Get-Content "$out\bin_vulkan.json" -Encoding UTF8 | ConvertFrom-Json).bin_dir
Write-Host "rocm  : $rocmBin"
Write-Host "vulkan: $vulkanBin"

# The runtime under test, when one is supplied: a copy of the release with only
# amdhip64_7.dll replaced, so the two arms differ by one file and nothing else.
$patchedBin = ""
if ($PatchedDllUrl) {
    $patchedBin = "$Work\lcpp\rocm_patched"
    Copy-Item -Recurse -Force -LiteralPath $rocmBin -Destination $patchedBin
    $member = ""
    if ($PatchedDllUrl -match "\.zip($|\?)") { $member = "amdhip64_7.dll" }
    & $py "$here\fetch.py" --url $PatchedDllUrl --dest "$patchedBin\amdhip64_7.dll" `
        --zip-member "$member" --expect-sha256 $PatchedDllSha256 --out "$out\patched_dll.json"
    if ($LASTEXITCODE -ne 0) { Fail "the runtime under test did not download, or did not match its hash" }
    Write-Host "patched HIP runtime staged at $patchedBin"
}

# --- 3. what every layer reports --------------------------------------------
Write-Host ""
Write-Host "== 3. what every layer reports =============================================="
& $py "$here\probe.py" --state stock --checkout $rocmBin --vulkan-dir $vulkanBin `
    --sections host,hip,vulkan_raw,ggml_vulkan --out "$out\memreport_stock.json"
if ($LASTEXITCODE -ne 0) { Write-Host "the memory report exited non-zero; the JSON still holds whatever it reached" }

# --- 4. the largest single allocation ---------------------------------------
# Bounded at 92% of the machine on purpose. The patched runtime's own note warns
# that committing near all of RAM can hang the host or bugcheck it (0x19C), and
# the question is whether an allocation the shipped runtime refuses now succeeds,
# not what the new ceiling is.
$ceil = [math]::Round($machine * 0.92, 1)
Write-Host ""
Write-Host "== 4. largest single hipMalloc that writes and reads back (search stops at $ceil GiB) =="
& $py "$here\probe.py" --state stock --checkout $rocmBin --sections host,hip,alloc_cap `
    --alloc-lo-gib 1 --alloc-hi-gib $ceil --alloc-resolution-gib 0.5 --alloc-reps 2 `
    --out "$out\alloccap_stock.json"
if ($patchedBin) {
    & $py "$here\probe.py" --state patched --checkout $patchedBin --sections host,hip,alloc_cap `
        --alloc-lo-gib 1 --alloc-hi-gib $ceil --alloc-resolution-gib 0.5 --alloc-reps 2 `
        --out "$out\alloccap_patched.json"
}

# --- 5. what Unsloth Studio would see ---------------------------------------
# Studio's own probe verbatim at a pinned commit. A re-implementation here would
# answer a different question: the claim is about what Studio sees.
Write-Host ""
Write-Host "== 5. Unsloth Studio's own Vulkan probe ====================================="
& $py "$here\fetch.py" `
    --url "https://raw.githubusercontent.com/unslothai/unsloth/ca42c015f/studio/backend/core/inference/_vulkan_probe.py" `
    --dest "$out\_vulkan_probe.py" --out "$out\studio_probe_source.json"
if ($LASTEXITCODE -eq 0) {
    & $py "$out\_vulkan_probe.py" $vulkanBin 2>&1 | Tee-Object -FilePath "$out\studio_vulkan_probe.txt"
    Write-Host "(index, free bytes, is_igpu, total bytes, name)"
} else {
    Write-Host "Studio's probe could not be fetched; this cell has no reading."
}

# --- 6. what llama.cpp would place, for a model you already have -------------
# llama-fit-params answers the placement question without serving, so this needs
# no port, no lifecycle and no generation: it reports what --fit decides against
# the numbers cell 3 measured, which is where an over-report turns into a model
# that will not load.
if ($ModelPath) {
    Write-Host ""
    Write-Host "== 6. what --fit would place ================================================"
    if (-not (Test-Path $ModelPath)) { Fail "-ModelPath does not exist: $ModelPath" }
    & $py "$here\probe.py" --state fit_rocm   --checkout $rocmBin   --sections fit --fit-model $ModelPath --out "$out\fit_rocm.json"
    & $py "$here\probe.py" --state fit_vulkan --checkout $vulkanBin --sections fit --fit-model $ModelPath --out "$out\fit_vulkan.json"
    foreach ($f in @("fit_rocm", "fit_vulkan")) {
        $j = Get-Content "$out\$f.json" -Encoding UTF8 | ConvertFrom-Json
        Write-Host "-- $f"
        Write-Host ($j.sections.fit.list_devices.stdout)
        if ($j.sections.fit.fit_params) {
            Write-Host ($j.sections.fit.fit_params.stdout)
            Write-Host ($j.sections.fit.fit_params.stderr)
        } else {
            Write-Host "   llama-fit-params is not in this build, so only --list-devices was read."
        }
    }
} else {
    Write-Host ""
    Write-Host "== 6. skipped: pass -ModelPath <first shard> to also ask what --fit would place"
}

# --- 7. the report ----------------------------------------------------------
Write-Host ""
Write-Host "== the three statements ====================================================="
$reports = @("stock=$out\memreport_stock.json")
& $py "$here\report.py" @reports 2>&1 | Tee-Object -FilePath "$out\REPORT.md"

Write-Host ""
Write-Host "Done. Send back this whole directory:" -ForegroundColor Green
Write-Host "  $out"
Write-Host "It holds REPORT.md plus the raw JSON every number came from."
