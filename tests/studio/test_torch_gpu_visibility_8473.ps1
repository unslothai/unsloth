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
$seen = Invoke-Vis "UNSLOTHTORCHGPU=1|2|2.9.0+rocm6.4|6.4.43482"
Check "an answer is an answer"    ($seen.Answered)
Check "the GPU is visible"        ($seen.SeesGpu)
Check "device count"              ($seen.DeviceCount -eq 2)
Check "torch version"             ($seen.TorchVersion -eq "2.9.0+rocm6.4")
Check "torch.version.hip"         ($seen.Hip -eq "6.4.43482")

$blind = Invoke-Vis "UNSLOTHTORCHGPU=0|0|2.9.0+cpu|"
Check "the mismatch is an answer too" ($blind.Answered)
Check "the GPU is not visible"        (-not $blind.SeesGpu)
Check "a cpu wheel has no hip"        ($blind.Hip -eq "")
Check "the cpu wheel is named"        ($blind.TorchVersion -eq "2.9.0+cpu")

Write-Host "and only the sentinel line is the answer"
# torch prints to stdout on some hosts; an unanchored match would read the banner.
$spoof = Invoke-Vis "warning: overriding UNSLOTHTORCHGPU=1|8|2.9.0|6.4`nUNSLOTHTORCHGPU=0|0|2.9.0+cpu|"
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
Check "it names the venv"                 ($report -match 'torch\.cuda\.is_available\(\) is False in \$VenvDir')
Check "it names the wheel"                ($report -match 'substep "torch \$\(\$_gpuVisibility\.TorchVersion\)')
Check "it names torch.version.hip"        ($report -match 'torch\.version\.hip \$_gpuCheckHip')
# Naming the symptom is what stops the user filing it a second time as a UI bug.
Check "it names what the user will see"   ($report -match 'No visible GPU')
Check "it says where to report it"        ($report -match 'github\.com/unslothai/unsloth/issues')
# A CPU-only Studio still chats. Failing the install over a diagnostic would be a regression.
Check "it never fails the setup"          (-not ($report -match 'Exit-SetupFailure|exit 1|\$stackExit'))
Check "a silent probe warns instead"      ($report -match '\[WARN\] could not check whether PyTorch sees this GPU')
# A GGUF-only venv has no torch and nothing to reconcile; warning there is noise every update.
Check "a missing torch is not warned about" ($report -match "ModuleNotFoundError\|No module named")

Write-Host "and it costs nothing where it cannot help"
Check "no-torch mode is excluded"         ($report -match '-not \$NoTorchMode')
Check "a CPU-only host is excluded"       ($report -match '\$_gpuCheckAnnounced -and')
Check "there is an escape hatch"          ($report -match 'UNSLOTH_SKIP_TORCH_GPU_CHECK')
Check "a missing interpreter is excluded" ($report -match 'Test-Path -LiteralPath \$_gpuCheckPy')
Check "it probes the venv interpreter"    ($report -match '\$_gpuCheckPy = Join-Path \$VenvDir "Scripts\\python\.exe"')
Check "the probe is called once"          ((([regex]::Matches($report, 'Get-TorchGpuVisibility')).Count) -eq 1)

Write-Host "the announced GPU is quoted back, whichever it was"
Check "NVIDIA is named"                   ($report -match 'if \(\$HasNvidiaSmi\) \{ \$_gpuCheckAnnounced = "NVIDIA GPU" \}')
Check "the AMD arch is named"             ($report -match '\$_gpuCheckAnnounced = "AMD GPU \(\$script:ROCmGfxArch\)"')
Check "an arch-less AMD host still counts" ($report -match 'elseif \(\$HasROCm\) \{ \$_gpuCheckAnnounced = "AMD GPU" \}')

# A hybrid Intel/NVIDIA host on the XPU wheel answers SeesGpu=False and still runs on the
# GPU, because _detect_hardware_locked falls through from CUDA to XPU. Accusing that host of
# running on CPU is a false alarm about a working machine (#8473). Test-VenvTorchIsXpu reads
# torch/version.py off disk, so the suppression cannot hang on a stalled Arc driver.
Check "the XPU suppression is present"    ($report -match 'Test-VenvTorchIsXpu')
Check "it suppresses, not merely reports" ($report -match 'not \(Test-VenvTorchIsXpu')
Check "it guards the mismatch arm"        ($report -match '\$_gpuVisibility\.SeesGpu -and[\s\S]{0,120}Test-VenvTorchIsXpu')

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
