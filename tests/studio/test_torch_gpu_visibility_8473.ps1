#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Issue #8473: the installer announces a GPU and the backend then runs CPU-only, with nothing
# reconciling the two verdicts. setup.ps1 now asks torch itself, after the dependency step, and
# prints the mismatch.
#
# Get-TorchGpuVisibility is driven with a stubbed Invoke-BoundedPythonProbe: the real one starts a
# process, and the answers that matter here (a banner ahead of the sentinel, a timeout, a crash)
# are exactly the ones a real interpreter will not produce on demand. There is no AMD hardware on
# any runner, so the report block itself is checked as wiring, on the real source text.
# Run: pwsh -NoProfile -File tests/studio/test_torch_gpu_visibility_8473.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setup = Join-Path $repo "studio/setup.ps1"

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

function Get-FunctionText {
    param([string] $Path, [string] $Name)
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$Path has parse errors" }
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $Name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $Name in $Path, found $($fn.Count)" }
    return $fn[0].Extent.Text
}

$visFn = Get-FunctionText $setup "Get-TorchGpuVisibility"
$leafFn = Get-FunctionText $setup "Get-TorchIndexLeaf"
# An empty or wrong extraction would make every case below pass vacuously.
Check "extraction kept the sentinel"  ($visFn -match 'UNSLOTHTORCHGPU')
Check "extraction kept the interpreter call" ($visFn -match 'Invoke-BoundedPythonProbe')

# --- Get-TorchGpuVisibility ------------------------------------------------------------------
# $Output/$Ok/$Error stand in for what the bounded probe returns; the stub also records the code
# it was handed, so the probe's own shape can be asserted without launching python.
function Invoke-Vis {
    param([string] $Output, [bool] $Ok = $true, [string] $ProbeError = "")
    $sb = [scriptblock]::Create(@"
param(`$Output, `$Ok, `$ProbeError)
function Invoke-BoundedPythonProbe {
    param([string]`$PythonExe, [string]`$Code, [int]`$TimeoutSec = 30)
    Set-Content -LiteralPath "$($script:codeSink)" -Value "`$TimeoutSec``n`$Code"
    return [pscustomobject]@{ Ok = `$Ok; Output = `$Output; Error = `$ProbeError }
}
$visFn
Get-TorchGpuVisibility -PythonExe 'C:\v\Scripts\python.exe'
"@)
    return (& $sb $Output $Ok $ProbeError)
}
$script:codeSink = Join-Path ([System.IO.Path]::GetTempPath()) "unsloth_8473_probe_code.txt"

Write-Host "torch's own answer is read, in full"
$seen = Invoke-Vis "UNSLOTHTORCHGPU=1|2|2.9.0+rocm6.4|6.4.43482|0"
Check "an answer is an answer"    ($seen.Answered)
Check "the GPU is visible"        ($seen.SeesGpu)
Check "device count"              ($seen.DeviceCount -eq 2)
Check "torch version"             ($seen.TorchVersion -eq "2.9.0+rocm6.4")
Check "torch.version.hip"         ($seen.Hip -eq "6.4.43482")

$blind = Invoke-Vis "UNSLOTHTORCHGPU=0|0|2.9.0+cpu||0"
Check "the mismatch is an answer too" ($blind.Answered)
Check "the GPU is not visible"        (-not $blind.SeesGpu)
Check "a cpu wheel has no hip"        ($blind.Hip -eq "")
Check "the cpu wheel is named"        ($blind.TorchVersion -eq "2.9.0+cpu")

Write-Host "and only the sentinel line is the answer"
# torch prints to stdout on some hosts; an unanchored match would read the banner.
$spoof = Invoke-Vis "warning: overriding UNSLOTHTORCHGPU=1|8|2.9.0|6.4|1`nUNSLOTHTORCHGPU=0|0|2.9.0+cpu||0"
Check "a banner cannot spoof it"  ($spoof.Answered -and -not $spoof.SeesGpu)

Write-Host "a probe that did not answer accuses nobody"
# Answered=False and SeesGpu=False are different facts. Collapsing them prints an accusation the
# run cannot support, on a host whose GPU may be perfectly fine.
$timedOut = Invoke-Vis "" $false "python did not answer within 90 seconds"
Check "a timeout is not an answer"    (-not $timedOut.Answered)
Check "a timeout is not a verdict"    (-not $timedOut.SeesGpu)
Check "the reason survives"           ($timedOut.Error -match "did not answer")
$crashed = Invoke-Vis "" $false "ModuleNotFoundError: No module named 'torch'"
Check "a crash is not an answer"      (-not $crashed.Answered)
Check "the crash text survives"       ($crashed.Error -match "ModuleNotFoundError")
# Ok with unparseable output is its own case: exit 0 plus a garbled line must not read as a verdict.
$garbled = Invoke-Vis "UNSLOTHTORCHGPU=maybe"
Check "garbled output is not an answer" (-not $garbled.Answered)
Check "a bare success is not an answer" (-not (Invoke-Vis "").Answered)

Write-Host "the probe itself is bounded, and asks all four questions"
$probeCode = Get-Content -Raw -LiteralPath $script:codeSink
Check "the bound is passed, not defaulted" ($probeCode -match '(?m)^90\s*$')
Check "it asks whether torch sees a GPU"   ($probeCode -match 'torch\.cuda\.is_available\(\)')
Check "it asks how many"                   ($probeCode -match 'torch\.cuda\.device_count\(\)')
Check "it asks which torch"                ($probeCode -match 'torch\.__version__')
Check "it asks which hip"                  ($probeCode -match "getattr\(torch\.version, 'hip'")
# Invoke-BoundedPythonProbe wraps the code in double quotes, so one inside would truncate it.
Check "the code carries no double quote"   (-not ($probeCode -match '"'))
Remove-Item -LiteralPath $script:codeSink -ErrorAction SilentlyContinue

# --- wiring ------------------------------------------------------------------------------------
# Normalised to LF once, rather than per pattern: on a CRLF checkout every \n-anchored pattern
# below matches nothing, and the -not checks over an empty region pass vacuously.
$setupText = (Get-Content -Raw $setup) -replace "`r`n", "`n"
$reportPat = '(?s)(# ── Does PyTorch see the GPU this installer announced\? ──.*?\n\}\n)'
$report = if ($setupText -match $reportPat) { $Matches[1] } else { "" }
Check "the report block was found"        ($report -ne "")
# Comment-stripped, for the checks that assert a variable is NOT read: the block's own comments
# name the flags it deliberately stopped re-deriving, and would satisfy those matches on prose.
$reportCode = (($report -split "`n") | Where-Object { $_ -notmatch '^\s*#' }) -join "`n"
Check "the comment strip left code"       ($reportCode -match 'Get-TorchGpuVisibility')
Check "CRLF is normalised, not tolerated" (-not (($setupText -replace "`n", "`r`n") -match $reportPat))

Write-Host "the check runs on the update that prints 'dependencies up to date'"
# The reported run took the fast path, so a check nested inside the dependency-pass branch would
# stay silent on exactly the run being complained about.
$_fastPathLine = ($setupText -split "`n" | Select-String -Pattern 'step "python" "dependencies up to date"' | Select-Object -First 1).LineNumber
$_checkLine = ($setupText -split "`n" | Select-String -Pattern '# ── Does PyTorch see the GPU this installer announced' | Select-Object -First 1).LineNumber
Check "it follows the fast-path message"  ($_fastPathLine -and $_checkLine -and $_fastPathLine -lt $_checkLine)
# ...and outside its else-block: the closing brace of the deps gate must sit between them.
$_gateClose = ($setupText -split "`n" | Select-String -Pattern '^\}$' | Where-Object { $_.LineNumber -gt $_fastPathLine } | Select-Object -First 1).LineNumber
Check "it is outside the deps gate"       ($_gateClose -and $_gateClose -lt $_checkLine)

Write-Host "it reports, and never aborts"
Check "the mismatch is an error line"     ($report -match 'step "gpu check" "PyTorch cannot see the \$_gpuCheckAnnounced reported above" "Red"')
Check "it names the venv"                 ($report -match '"\$_gpuCheckApi in \$VenvDir"')
# Only an Intel announcement is held back until BOTH answers are False, so only there may the
# report say so; naming torch.xpu on an NVIDIA/AMD host would state something never checked.
Check "an Intel report names both answers" ($report -match 'Intel\*[\s\S]{0,120}torch\.xpu\.is_available\(\) are False')
Check "every other report names torch.cuda only" ($report -match 'else \{ "torch\.cuda\.is_available\(\) is False" \}')
Check "it names the wheel"                ($report -match 'substep "torch \$\(\$_gpuVisibility\.TorchVersion\)')
Check "it names torch.version.hip"        ($report -match 'torch\.version\.hip \$_gpuCheckHip')
# Naming the symptom is what stops the user filing it a second time as a UI bug.
Check "it names what the user will see"   ($report -match 'No visible GPU')
# ...conditionally. llama.cpp is a separate stack: with the Vulkan bundle the backend fills
# inference_gpu from get_vulkan_inference_gpu_info() and the monitor shows that card's real
# VRAM, so promising "--" and a CPU-only Studio would be a false prediction there.
Check "it does not promise a CPU-only Studio" (-not ($report -match 'Studio will run CPU-only'))
# hardware.py leaves CHAT_ONLY true on the CPU fallback and disables Train/Export, so
# "training will run on CPU" was the opposite of what happens. Same wording as the XPU arm.
Check "it claims only what torch answered" ($report -match 'PyTorch training and GPU inference are unavailable; chat and GGUF still work')
# "PyTorch", because a false torch.cuda.is_available() says nothing about llama.cpp: a CUDA /
# HIP / Vulkan GGUF bundle still offloads to the same card.
Check "the claim is scoped to torch"      (-not ($report -match '(?<!PyTorch )training and GPU inference are unavailable'))
Check "it does not promise CPU training" (-not ($report -match 'will run on CPU'))
Check "the monitor line is conditional"   ($report -match 'If the Live monitor shows VRAM')
Check "chat and GGUF are exempted"        ($report -match 'chat and GGUF still work')
Check "it says where to report it"        ($report -match 'github\.com/unslothai/unsloth/issues')
# A CPU-only Studio still chats. Failing the install over a diagnostic would be a regression.
Check "it never fails the setup"          (-not ($report -match 'Exit-SetupFailure|exit 1|\$stackExit'))
Check "a silent probe warns instead"      ($report -match '\[WARN\] could not check whether PyTorch sees this GPU')
# A GGUF-only venv has no torch and nothing to reconcile; warning there is noise every update.
Check "a missing torch is not warned about" ($report -match "No module named 'torch'")

Write-Host "the quiet-when-torch-is-absent arm is about torch, and only torch"
# Run the real arm rather than reading it: a match broad enough to swallow any import error
# silences the host this check exists for, and the source text alone cannot show that.
$quietPat = '(?ms)^        if \(-not \(\$_gpuVisibility\.Error -match.*?^        \}$'
$quietArm = if ($report -match $quietPat) { $Matches[0] } else { "" }
Check "the quiet arm was found"           ($quietArm -ne "")
function Test-Warns {
    param([string] $ErrText)
    $sb = [scriptblock]::Create(@"
param(`$ErrText)
`$script:Warned = `$false
function substep { param(`$a, `$b) `$script:Warned = `$true }
`$_gpuVisibility = [pscustomobject]@{ Answered = `$false; Error = `$ErrText }
$quietArm
`$script:Warned
"@)
    return (& $sb $ErrText)
}
Check "an absent torch says nothing"      (-not (Test-Warns "ModuleNotFoundError: No module named 'torch'"))
# The backend treats ANY torch import failure as detection failure and runs on CPU, so a torch
# that is installed and cannot import is exactly the host that needs telling.
Check "a missing transitive dep warns"    (Test-Warns "ModuleNotFoundError: No module named 'typing_extensions'")
Check "a broken torch internal warns"     (Test-Warns "ModuleNotFoundError: No module named 'torch._C'")
Check "a timeout still warns"             (Test-Warns "python did not answer within 90 seconds")
Check "an OSError still warns"            (Test-Warns "OSError: [WinError 126] The specified module could not be found")

Write-Host "and it costs nothing where it cannot help"
Check "no-torch mode is excluded"         ($report -match '-not \$NoTorchMode')
Check "a CPU-only host is excluded"       ($report -match '\$_gpuCheckAnnounced -and')
Check "there is an escape hatch"          ($report -match 'UNSLOTH_SKIP_TORCH_GPU_CHECK')
# An explicit cpu pin is a request, not a fault: install_python_stack force-reinstalls the CPU
# wheel for it, so a red "cannot see the GPU" would accuse the user of their own setting.
Check "an explicit cpu pin is excluded"   ($report -match '\$_gpuCheckPinLeaf -ne "cpu"')
# A hide-all visibility mask is a request too, and detection cannot see it: nvidia-smi ignores
# CUDA_VISIBLE_DEVICES, and the AMD arch can come from a WMI GPU name, so the summary announces
# the card while torch is meant to see nothing.
Check "a hide-all mask is excluded"       ($report -match '-not \$_gpuCheckMasked')
# Scoped to the mask that governs what was announced: an idle HIP mask on an NVIDIA host must
# not mute a genuine mismatch.
Check "the NVIDIA arm reads the CUDA mask" ($report -match 'NVIDIA\*.*[\s\S]{0,80}CUDA_VISIBLE_DEVICES')
# All three, in the order hardware.py resolves them: ROCm layers HIP/ROCR on top of the CUDA
# mask and falls back to it when neither is set.
Check "the AMD arm reads the HIP masks"   ($report -match 'HIP_VISIBLE_DEVICES.*ROCR_VISIBLE_DEVICES.*CUDA_VISIBLE_DEVICES')
Check "the NVIDIA arm ignores HIP"        (-not ($report -match 'NVIDIA\*\) \{\s*\n\s*Test-VisibleMaskHidesAll \$env:HIP'))

# The mask predicate itself is run: PowerShell turns an unset $env: read into "" under a
# [string[]] cast, and "" is the hide-all value, so a typed parameter would mute every host.
$maskFn = Get-FunctionText $setup "Test-VisibleMaskHidesAll"
function Test-Mask {
    param($Masks)
    $sb = [scriptblock]::Create(@"
param(`$Masks)
$maskFn
Test-VisibleMaskHidesAll `$Masks
"@)
    return (& $sb $Masks)
}
Check "an unset mask hides nothing"       (-not (Test-Mask $null))
Check "no mask at all hides nothing"      (-not (Test-Mask @($null, $null)))
Check "-1 hides everything"               (Test-Mask "-1")
Check "an empty mask hides everything"    (Test-Mask "")
Check "whitespace is trimmed"             (Test-Mask " -1 ")
Check "a selected device is not hidden"   (-not (Test-Mask "0"))
Check "a device list is not hidden"       (-not (Test-Mask "1,0"))
# First-set-wins: an empty HIP mask shadows a ROCR mask that names a device.
Check "the first set mask wins"           (Test-Mask @("", "0"))
Check "and an unset one is skipped"       (-not (Test-Mask @($null, "0")))
Check "a bare CUDA mask still hides"      (Test-Mask @($null, $null, "-1"))
Check "a named AMD device wins over it"   (-not (Test-Mask @("0", $null, "-1")))
# Resolved here, not read off $TorchIndexPinned / $CuTag: those live inside the dependency-pass
# branch and are $null on the "dependencies up to date" run this check exists for.
Check "the pin is resolved fresh"         ($report -match '\$_gpuCheckPinLeaf = Get-TorchIndexLeaf \(Get-PinnedTorchIndexUrl\)')
Check "a GPU pin is not excluded with it" (-not ($reportCode -match '\$CuTag|\$TorchIndexPinned'))
Check "a missing interpreter is excluded" ($report -match 'Test-Path -LiteralPath \$_gpuCheckPy')
Check "it probes the venv interpreter"    ($report -match '\$_gpuCheckPy = Join-Path \$VenvDir "Scripts\\python\.exe"')
Check "the probe is called once"          ((([regex]::Matches($report, 'Get-TorchGpuVisibility')).Count) -eq 1)

Write-Host "the announced GPU is quoted from the summary, not re-derived"
# Re-deriving from the raw detection flags disagreed with the summary on a host with an Intel
# Arc next to an AMD card whose arch gets no ROCm wheels: the scan lets Intel win there and
# $script:ROCmGfxArch stays populated, so the report named a GPU nobody announced (#8473).
Check "it quotes the summary"             ($reportCode -match '\$_gpuCheckAnnounced = \$script:GpuSummaryAnnounced')
Check "it re-derives nothing"             (-not ($reportCode -match '\$_gpuCheckAnnounced = "'))
Check "it reads no detection flag"        (-not ($reportCode -match '\$script:ROCmGfxArch|\$HasROCm|\$HasNvidiaSmi|\$script:IsIntelXpu'))

# ...and the summary itself records what it printed. Run the real chain, so an announcement that
# stops being recorded fails here rather than passing on the report's source text alone.
$summaryPat = '(?ms)^\$script:GpuSummaryAnnounced = \$null\nif \(\$HasNvidiaSmi\) \{.*?^\}$'
$summary = if ($setupText -match $summaryPat) { $Matches[0] } else { "" }
Check "the GPU summary was found"         ($summary -ne "")
Check "the extraction kept every arm"     ($summary -match 'Intel GPU detected' -and $summary -match 'AMD ROCm \(\$script:ROCmGfxArch\)' -and $summary -match 'none \(chat-only / GGUF\)')

function Invoke-Summary {
    param([hashtable] $Vars)
    $sb = [scriptblock]::Create(@"
param(`$V)
`$script:Lines = @()
function step { param(`$a, `$b, `$c) `$script:Lines += "STEP|`$a|`$b" }
function substep { param(`$a, `$b) `$script:Lines += "SUB|`$a" }
function Write-StudioLine { param(`$a, `$b, `$c) }
`$HasNvidiaSmi = [bool]`$V['HasNvidiaSmi']
`$script:IsIntelXpu = [bool]`$V['IsIntelXpu']
`$IntelGpuLabel = `$V['IntelGpuLabel']
`$HasROCm = [bool]`$V['HasROCm']
`$ROCmGpuLabel = `$V['ROCmGpuLabel']
`$HipSdkInstalled = [bool]`$V['HipSdkInstalled']
`$AmdHasGpuWheels = [bool]`$V['AmdHasGpuWheels']
`$_amdPinIsGpu = [bool]`$V['AmdPinIsGpu']
`$script:ROCmGfxArch = `$V['ROCmGfxArch']
`$script:ROCmVersionFull = `$V['ROCmVersionFull']
$summary
[pscustomobject]@{ Announced = `$script:GpuSummaryAnnounced; Lines = (`$script:Lines -join "``n") }
"@)
    return (& $sb $Vars)
}

$sNvidia = Invoke-Summary @{ HasNvidiaSmi = $true; ROCmGfxArch = "gfx1201" }
Check "NVIDIA is announced as NVIDIA"     ($sNvidia.Announced -eq "NVIDIA GPU")
Check "and that is what it printed"       ($sNvidia.Lines -match 'STEP\|gpu\|NVIDIA GPU detected')

# The reported combination: Intel Arc plus an AMD card on an arch outside $_rocmWheelArches, so
# the Intel scan runs, Intel wins the summary, and $script:ROCmGfxArch is still set.
$sHybrid = Invoke-Summary @{ IsIntelXpu = $true; IntelGpuLabel = "Intel Arc B580"; ROCmGfxArch = "gfx90c" }
Check "Intel wins the summary"            ($sHybrid.Lines -match 'STEP\|gpu\|Intel GPU detected')
Check "and Intel is what is announced"    ($sHybrid.Announced -eq "Intel GPU")
Check "no AMD announcement is invented"   (-not ($sHybrid.Announced -match 'AMD'))

$sAmd = Invoke-Summary @{ ROCmGfxArch = "gfx1201"; AmdHasGpuWheels = $true }
Check "a ROCm-wheel AMD host is announced" ($sAmd.Announced -eq "AMD GPU (gfx1201)")
$sHip = Invoke-Summary @{ HasROCm = $true; ROCmGpuLabel = "AMD ROCm (gfx1100)"; ROCmGfxArch = "gfx1100"; AmdHasGpuWheels = $true }
Check "a HIP SDK AMD host is announced"   ($sHip.Announced -eq "AMD GPU (gfx1100)")

# Nothing to reconcile: no accelerator at all, and any AMD card the install path deliberately
# sends to CPU torch. A red "PyTorch cannot see it" there contradicts the line above it and asks
# the user to report a working configuration. Every AMD arm has to agree, so all three are run.
$sNone = Invoke-Summary @{}
Check "a CPU-only host announces nothing" ($null -eq $sNone.Announced)
$sUnknownArch = Invoke-Summary @{ ROCmGpuLabel = "AMD Radeon 780M" }
Check "an arch-less AMD host is not accused" ($null -eq $sUnknownArch.Announced)
# Vega / RDNA1 / MI300 on Windows: $HasROCm is true, the arch is outside $_rocmWheelArches, and
# the install path warns "not in supported arch list -- falling back to CPU-only PyTorch".
$sVega = Invoke-Summary @{ HasROCm = $true; ROCmGpuLabel = "AMD ROCm (gfx900)"; ROCmGfxArch = "gfx900" }
Check "an unmapped arch on the ROCm arm is not accused" ($null -eq $sVega.Announced)
Check "and the summary still announced it"  ($sVega.Lines -match 'STEP\|gpu\|AMD ROCm \(gfx900\)')
$sVegaSdk = Invoke-Summary @{ HipSdkInstalled = $true; ROCmGpuLabel = "AMD Radeon VII"; ROCmGfxArch = "gfx906" }
Check "an unmapped arch on the HIP SDK arm is not accused" ($null -eq $sVegaSdk.Announced)
$sVegaBundled = Invoke-Summary @{ ROCmGfxArch = "gfx1010" }
Check "an unmapped arch on the bundled-wheel arm is not accused" ($null -eq $sVegaBundled.Announced)

# ...unless a pin overrides the arch decision. The pinned ROCm path routes a gfx*/rocm* leaf
# through the GPU index whatever the arch and skips the CPU fallback, so that host really does
# get a GPU wheel and a torch that cannot open it is worth reporting.
$sPinned = Invoke-Summary @{ HasROCm = $true; ROCmGpuLabel = "AMD ROCm (gfx1010)"; ROCmGfxArch = "gfx1010"; AmdPinIsGpu = $true }
Check "a pinned unmapped arch is reconciled" ($sPinned.Announced -eq "AMD GPU (gfx1010)")
$sPinnedSdk = Invoke-Summary @{ HipSdkInstalled = $true; ROCmGpuLabel = "AMD Radeon VII"; ROCmGfxArch = "gfx906"; AmdPinIsGpu = $true }
Check "the HIP SDK arm honours a pin too"   ($sPinnedSdk.Announced -eq "AMD GPU (gfx906)")
$sPinnedBundled = Invoke-Summary @{ ROCmGfxArch = "gfx942"; AmdPinIsGpu = $true }
Check "the bundled-wheel arm honours a pin too" ($sPinnedBundled.Announced -eq "AMD GPU (gfx942)")
# The arch-unknown arm has no gfx to name, but a pin still installs a GPU wheel there.
$sPinnedNoArch = Invoke-Summary @{ ROCmGpuLabel = "AMD Radeon 780M"; AmdPinIsGpu = $true }
Check "an arch-less pinned host is reconciled" ($sPinnedNoArch.Announced -eq "AMD GPU")
Check "and it is not named 'AMD GPU ()'"    (-not ($sPinnedNoArch.Announced -match '\(\)'))
# The pin predicate itself is run, not read: `$null -ne "cpu"` is TRUE in PowerShell, so the
# emptiness half is what stops every unpinned host announcing again.
$pinPat = '(?ms)^\$_amdPinLeaf = Get-TorchIndexLeaf.*?\n\$_amdPinIsGpu = .*?$'
$pinExpr = if ($setupText -match $pinPat) { $Matches[0] } else { "" }
Check "the pin predicate was found"       ($pinExpr -ne "")
function Test-AmdPinIsGpu {
    param([string] $Url)
    $sb = [scriptblock]::Create(@"
param(`$Url)
function Get-PinnedTorchIndexUrl { if ([string]::IsNullOrWhiteSpace(`$Url)) { return `$null } return `$Url }
$leafFn
$pinExpr
`$_amdPinIsGpu
"@)
    return (& $sb $Url)
}
Check "no pin is not a GPU pin"           (-not (Test-AmdPinIsGpu ""))
Check "a cpu pin is not a GPU pin"        (-not (Test-AmdPinIsGpu "https://download.pytorch.org/whl/cpu"))
Check "a gfx pin is a GPU pin"            (Test-AmdPinIsGpu "https://repo.amd.com/rocm/whl/gfx1010/")
Check "a rocm pin is a GPU pin"           (Test-AmdPinIsGpu "https://download.pytorch.org/whl/rocm7.1")
Check "a cuda pin is a GPU pin"           (Test-AmdPinIsGpu "https://download.pytorch.org/whl/cu128")

# A hybrid Intel/NVIDIA host on the XPU wheel answers SeesGpu=False and still runs on the
# GPU, because _detect_hardware_locked falls through from CUDA to XPU. Accusing that host of
# running on CPU is a false alarm about a working machine (#8473). Test-VenvTorchIsXpu reads
# probe, so a +xpu wheel whose Intel runtime is dead is still reported.
Check "the XPU suppression is present"    ($report -match 'SeesXpu')
Check "it suppresses, not merely reports" ($report -match 'not \$_gpuVisibility\.SeesXpu')
Check "it guards the mismatch arm"        ($report -match '\$_gpuVisibility\.SeesGpu -and[\s\S]{0,120}SeesXpu')
Check "the suppression reads the probe, not the wheel label" (-not ($report -match 'Test-VenvTorchIsXpu'))

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
