#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Issue #8473: the installer announces a GPU and the backend then runs CPU-only, with nothing
# reconciling the two verdicts. Get-TorchGpuVisibility is driven with a stubbed
# Invoke-BoundedPythonProbe: a banner, a timeout and a crash are not producible on demand.
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
# Defined up here, not beside its own cases: the mask-arm block above them runs the real
# $_gpuCheckMasked chain, whose Intel arm calls this.
$oneApiFn = Get-FunctionText $setup "Test-OneApiSelectorExcludesGpu"
Invoke-Expression $oneApiFn
# An empty or wrong extraction would make every case below pass vacuously.
Check "extraction kept the sentinel"  ($visFn -match 'UNSLOTHTORCHGPU')
Check "extraction kept the interpreter call" ($visFn -match 'Invoke-BoundedPythonProbe')

# --- Get-TorchGpuVisibility ------------------------------------------------------------------
# The stub records the code it was handed, so the probe's shape is asserted without launching python.
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
$spoof = Invoke-Vis "warning: overriding UNSLOTHTORCHGPU=1|8|2.9.0|6.4|1`nUNSLOTHTORCHGPU=0|0|2.9.0+cpu||0"
Check "a banner cannot spoof it"  ($spoof.Answered -and -not $spoof.SeesGpu)

Write-Host "a probe that did not answer accuses nobody"
$timedOut = Invoke-Vis "" $false "python did not answer within 90 seconds"
Check "a timeout is not an answer"    (-not $timedOut.Answered)
Check "a timeout is not a verdict"    (-not $timedOut.SeesGpu)
Check "the reason survives"           ($timedOut.Error -match "did not answer")
$crashed = Invoke-Vis "" $false "ModuleNotFoundError: No module named 'torch'"
Check "a crash is not an answer"      (-not $crashed.Answered)
Check "the crash text survives"       ($crashed.Error -match "ModuleNotFoundError")
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
# Normalised to LF once: on a CRLF checkout every \n-anchored pattern below matches nothing, and
# the -not checks over an empty region pass vacuously.
$setupText = (Get-Content -Raw $setup) -replace "`r`n", "`n"
$reportPat = '(?s)(# ── Does PyTorch see the GPU this installer announced\? ──.*?\n\}\n)'
$report = if ($setupText -match $reportPat) { $Matches[1] } else { "" }
Check "the report block was found"        ($report -ne "")
# Comment-stripped for the checks that assert a variable is NOT read: the block's own comments name
# the flags it stopped re-deriving, and would satisfy those matches on prose.
$reportCode = (($report -split "`n") | Where-Object { $_ -notmatch '^\s*#' }) -join "`n"
Check "the comment strip left code"       ($reportCode -match 'Get-TorchGpuVisibility')
Check "CRLF is normalised, not tolerated" (-not (($setupText -replace "`n", "`r`n") -match $reportPat))

Write-Host "the check runs on the update that prints 'dependencies up to date'"
$_fastPathLine = ($setupText -split "`n" | Select-String -Pattern 'step "python" "dependencies up to date"' | Select-Object -First 1).LineNumber
$_checkLine = ($setupText -split "`n" | Select-String -Pattern '# ── Does PyTorch see the GPU this installer announced' | Select-Object -First 1).LineNumber
Check "it follows the fast-path message"  ($_fastPathLine -and $_checkLine -and $_fastPathLine -lt $_checkLine)
$_gateClose = ($setupText -split "`n" | Select-String -Pattern '^\}$' | Where-Object { $_.LineNumber -gt $_fastPathLine } | Select-Object -First 1).LineNumber
Check "it is outside the deps gate"       ($_gateClose -and $_gateClose -lt $_checkLine)

Write-Host "it reports, and never aborts"
Check "the mismatch is an error line"     ($report -match 'step "gpu check" "PyTorch cannot see the \$_gpuCheckAnnounced reported above" "Red"')
Check "it names the venv"                 ($report -match '"\$_gpuCheckApi in \$VenvDir"')
# Only an Intel announcement is held to BOTH answers, so naming torch.xpu elsewhere states nothing.
Check "an Intel report names both answers" ($report -match 'Intel\*[\s\S]{0,120}torch\.xpu\.is_available\(\) are False')
Check "every other report names torch.cuda only" ($report -match 'else \{ "torch\.cuda\.is_available\(\) is False" \}')
Check "it names the wheel"                ($report -match 'substep "torch \$\(\$_gpuVisibility\.TorchVersion\)')
Check "it names torch.version.hip"        ($report -match 'torch\.version\.hip \$_gpuCheckHip')
Check "it names what the user will see"   ($report -match 'No visible GPU')
# ...conditionally: with a Vulkan GGUF bundle the monitor shows real VRAM, so "--" would be false.
Check "it does not promise a CPU-only Studio" (-not ($report -match 'Studio will run CPU-only'))
# hardware.py leaves CHAT_ONLY true on the fallback and disables Train/Export.
Check "it claims only what torch answered" ($report -match 'PyTorch training and GPU inference are unavailable; chat and GGUF still work')
# A false torch.cuda.is_available() says nothing about llama.cpp: a GGUF bundle still offloads.
Check "the claim is scoped to torch"      (-not ($report -match '(?<!PyTorch )training and GPU inference are unavailable'))
Check "it does not promise CPU training" (-not ($report -match 'will run on CPU'))
Check "the monitor line is conditional"   ($report -match 'If the Live monitor shows VRAM')
Check "chat and GGUF are exempted"        ($report -match 'chat and GGUF still work')
Check "it says where to report it"        ($report -match 'github\.com/unslothai/unsloth/issues')
# A CPU-only Studio still chats, so failing the install over a diagnostic would be a regression.
Check "it never fails the setup"          (-not ($report -match 'Exit-SetupFailure|exit 1|\$stackExit'))
Check "a silent probe warns instead"      ($report -match '\[WARN\] could not check whether PyTorch sees this GPU')
# A GGUF-only venv has no torch and nothing to reconcile; warning there is noise every update.
Check "a missing torch is not warned about" ($report -match "No module named 'torch'")

Write-Host "the quiet-when-torch-is-absent arm is about torch, and only torch"
# Run rather than read: source text alone cannot show that the match is narrow enough.
$quietPat = '(?ms)^        if \(-not \(\$_gpuVisibility\.Error -match.*?^        \}$'
$quietArm = if ($report -match $quietPat) { $Matches[0] } else { "" }
Check "the quiet arm was found"           ($quietArm -ne "")
function Test-Warns {
    param([string] $ErrText)
    $sb = [scriptblock]::Create(@"
param(`$ErrText)
`$script:Lines = @()
function substep { param(`$a, `$b) `$script:Lines += `$a }
`$_gpuCheckPy = "C:\venv\Scripts\python.exe"
`$_gpuVisibility = [pscustomobject]@{ Answered = `$false; Error = `$ErrText }
$quietArm
,`$script:Lines
"@)
    return @(& $sb $ErrText)
}
Check "an absent torch says nothing"      ((Test-Warns "ModuleNotFoundError: No module named 'torch'").Count -eq 0)
# The backend treats ANY torch import failure as detection failure and runs on CPU, so an installed
# torch that cannot import is exactly the host that needs telling.
Check "a missing transitive dep warns"    ((Test-Warns "ModuleNotFoundError: No module named 'typing_extensions'").Count -gt 0)
Check "a broken torch internal warns"     ((Test-Warns "ModuleNotFoundError: No module named 'torch._C'").Count -gt 0)
Check "a timeout still warns"             ((Test-Warns "python did not answer within 90 seconds").Count -gt 0)
Check "an OSError still warns"            ((Test-Warns "OSError: [WinError 126] The specified module could not be found").Count -gt 0)

# The reason is quoted from the LAST non-empty line only. Interpolating the whole .Error put a
# raw multi-line traceback through `substep`, which pads only its first line -- and line two of a
# CPython SyntaxError traceback is the entire 250-character probe source. The last line is the one
# worth keeping: CPython puts the exception type and message there, while the first line is the
# fixed "Traceback (most recent call last):" banner that says nothing about what went wrong.
$_multi = Test-Warns "Traceback (most recent call last):`n  File `"<string>`", line 1`n    import signal; signal.alarm(90); import torch`nOSError: [WinError 126]"
Check "the warning is not a traceback"    ($_multi.Count -eq 2)
Check "it quotes one line, the useful one" ($_multi[0] -match 'OSError: \[WinError 126\]$')
Check "it does not quote the banner"      (-not ($_multi[0] -match 'Traceback \(most recent call last\):'))
Check "it never leaks the probe source"   (-not ($_multi[0] -match 'signal\.alarm'))
Check "and carries no probe source"       (-not ($_multi -match 'signal\.alarm'))
Check "it says how to reproduce"          ($_multi[1] -match 'torch\.cuda\.is_available\(\)')
# A non-zero exit with no stderr, or stdout that misses the line anchor, left a WARN ending in a
# bare colon with nothing after it.
$_empty = Test-Warns ""
Check "an empty reason is not printed"    ($_empty.Count -eq 2)
Check "and leaves no dangling colon"      ($_empty[0] -notmatch ':\s*$')


Write-Host "and it costs nothing where it cannot help"
Check "no-torch mode is excluded"         ($report -match '-not \$NoTorchMode')
Check "a CPU-only host is excluded"       ($report -match '\$_gpuCheckAnnounced -and')
Check "there is an escape hatch"          ($report -match 'UNSLOTH_SKIP_TORCH_GPU_CHECK')
# An explicit cpu pin is a request, not a fault, so a red report would accuse the user's own setting.
Check "an explicit cpu pin is excluded"   ($report -match '\$_gpuCheckPinLeaf -ne "cpu"')
# A hide-all mask is a request too, and detection cannot see it: nvidia-smi ignores it entirely.
Check "a hide-all mask is excluded"       ($report -match '-not \$_gpuCheckMasked')
# install.ps1 routes an NVIDIA host whose CUDA is below 11 to the CPU index by design, but $null is
# "did not say", so a presence test would disable the check on every standalone update.
Check "an installer cpu tag is excluded"  ($report -match '\$InstallerTorchTag -ne "cpu"')
Check "an absent tag is not read as cpu"  (-not ($reportCode -match '-not \$InstallerTorchTag|\$InstallerTorchTag -eq \$null'))
# Scoped to the mask governing what was announced: an idle HIP mask must not mute a real mismatch.
Check "the NVIDIA arm reads the CUDA mask" ($report -match 'NVIDIA\*.*[\s\S]{0,80}CUDA_VISIBLE_DEVICES')
# HIP then CUDA, the two masks clr declares, in hardware.py's order.
Check "the AMD arm reads the HIP masks"   ($reportCode -match 'HIP_VISIBLE_DEVICES.*CUDA_VISIBLE_DEVICES')
# ROCR_VISIBLE_DEVICES belongs to the ROCr runtime, which Windows HIP does not have.
Check "the AMD arm ignores ROCR"          (-not ($reportCode -match 'ROCR_VISIBLE_DEVICES'))
Check "the NVIDIA arm ignores HIP"        (-not ($report -match 'NVIDIA\*\) \{\s*\n\s*Test-VisibleMaskHidesAll \$env:HIP'))

# The mask predicate itself is run: a [string[]] cast turns an unset $env: read into "", the
# hide-all value, so a typed parameter would mute every host.
$maskFn = Get-FunctionText $setup "Test-VisibleMaskHidesAll"
# The Intel arm of the chain calls both predicates, so both must be in scope when it runs.
$zeFn = Get-FunctionText $setup "Test-ZeAffinityMaskHidesAll"
Invoke-Expression $zeFn
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
# The runtime discards everything to the right of the first invalid entry, so a leading negative
# leaves nothing visible, while a trailing one still leaves the devices named before it.
Check "a leading negative hides all"      (Test-Mask "-1,0")
Check "any negative index hides all"      (Test-Mask "-2")
Check "a trailing negative is a selection" (-not (Test-Mask "0,-1"))
Check "the first set mask wins"           (Test-Mask @("", "0"))
Check "and an unset one is skipped"       (-not (Test-Mask @($null, "0")))
Check "a bare CUDA mask still hides"      (Test-Mask @($null, "-1"))
Check "a named AMD device wins over it"   (-not (Test-Mask @("0", "-1")))

Write-Host "and the arms that pick the masks are run, not merely read"
# Always-suppress is the failure mode that would hide #8473 again, and source text cannot tell it
# from a scoped exclusion, so the expression is executed. Unset via the Env: drive, never
# [Environment]::SetEnvironmentVariable($n, $null): PowerShell converts $null to "" when it binds a
# [string] parameter, leaving the variable SET and empty, which is the hide-all value.
$maskedPat = '(?ms)^\$_gpuCheckMasked = if \(.*?^\} else \{ \$false \}$'
$maskedExpr = if ($report -match $maskedPat) { $Matches[0] } else { "" }
Check "the mask arms were found"          ($maskedExpr -match 'Test-VisibleMaskHidesAll')
function Test-Arm {
    param([string] $Announced, [hashtable] $Vars)
    $names = @("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "ZE_AFFINITY_MASK")
    $saved = @{}
    foreach ($n in $names) {
        $saved[$n] = [Environment]::GetEnvironmentVariable($n)
        if (Test-Path "Env:\$n") { Remove-Item "Env:\$n" }
    }
    foreach ($n in $Vars.Keys) { Set-Item "Env:\$n" -Value $Vars[$n] }
    try {
        $sb = [scriptblock]::Create(@"
param(`$_gpuCheckAnnounced)
$maskFn
$oneApiFn
$zeFn
$maskedExpr
`$_gpuCheckMasked
"@)
        return [bool] (& $sb $Announced)
    } finally {
        foreach ($n in $names) {
            if (Test-Path "Env:\$n") { Remove-Item "Env:\$n" }
            if ($null -ne $saved[$n]) { Set-Item "Env:\$n" -Value $saved[$n] }
        }
    }
}
Check "a hidden AMD card is masked"       (Test-Arm "AMD GPU (gfx1201)" @{ HIP_VISIBLE_DEVICES = "-1" })
Check "a selected AMD card is not"        (-not (Test-Arm "AMD GPU (gfx1201)" @{ HIP_VISIBLE_DEVICES = "0" }))
Check "a bare CUDA mask hides AMD too"    (Test-Arm "AMD GPU (gfx1201)" @{ CUDA_VISIBLE_DEVICES = "-1" })
# The two halves of the Windows ROCR gate. A dead ROCR var must neither mute a broken torch...
Check "a lone ROCR mask hides nothing"    (-not (Test-Arm "AMD GPU (gfx1201)" @{ ROCR_VISIBLE_DEVICES = "-1" }))
# ...nor, by winning first-set-wins, shadow the CUDA mask that does hide the card.
Check "nor shadow a real CUDA hide"       (Test-Arm "AMD GPU (gfx1201)" @{ ROCR_VISIBLE_DEVICES = "0"; CUDA_VISIBLE_DEVICES = "-1" })
Check "a hidden NVIDIA card is masked"    (Test-Arm "NVIDIA GPU" @{ CUDA_VISIBLE_DEVICES = "-1" })
Check "a selected NVIDIA card is not"     (-not (Test-Arm "NVIDIA GPU" @{ CUDA_VISIBLE_DEVICES = "0" }))
Check "an idle HIP mask mutes nothing"    (-not (Test-Arm "NVIDIA GPU" @{ HIP_VISIBLE_DEVICES = "-1" }))
# Both Intel hides, through the real chain. An EMPTY ZE_AFFINITY_MASK stays a no-op; a non-empty
# one filters, and a negative entry can never name a root device.
Check "an empty ZE mask hides nothing"    (-not (Test-Arm "Intel GPU" @{ ZE_AFFINITY_MASK = "" }))
Check "'default' hides nothing"           (-not (Test-Arm "Intel GPU" @{ ZE_AFFINITY_MASK = "default" }))
Check "ZE=-1 hides every device"          (Test-Arm "Intel GPU" @{ ZE_AFFINITY_MASK = "-1" })
Check "a ZE ordinal is a selection"       (-not (Test-Arm "Intel GPU" @{ ZE_AFFINITY_MASK = "0" }))
Check "so is a ZE sub-device"             (-not (Test-Arm "Intel GPU" @{ ZE_AFFINITY_MASK = "0.1" }))
Check "so is a ZE list"                   (-not (Test-Arm "Intel GPU" @{ ZE_AFFINITY_MASK = "0,1" }))
Check "a mixed list is a selection"       (-not (Test-Arm "Intel GPU" @{ ZE_AFFINITY_MASK = "-1,0" }))
Check "an all-negative list hides all"    (Test-Arm "Intel GPU" @{ ZE_AFFINITY_MASK = "-1,-2" })
Check "garbage fails open"                (-not (Test-Arm "Intel GPU" @{ ZE_AFFINITY_MASK = "hello" }))
Check "the selector still hides too"      (Test-Arm "Intel GPU" @{ ONEAPI_DEVICE_SELECTOR = "*:cpu" })
# ...and neither Intel variable may leak into the other vendors' arms.
Check "a ZE mask mutes no NVIDIA host"    (-not (Test-Arm "NVIDIA GPU" @{ ZE_AFFINITY_MASK = "-1" }))
Check "a ZE mask mutes no AMD host"       (-not (Test-Arm "AMD GPU (gfx1201)" @{ ZE_AFFINITY_MASK = "-1" }))

# Called directly as well as through the chain: Windows PowerShell 5.1 cannot hold an empty
# environment variable (setting one deletes it), so the `Set-Item` route above can never deliver
# a set-but-empty mask and cannot see the early return that must keep it a no-op. A parent
# process can set one, and pwsh 7.5+ can too, so the branch is live and needs its own case.
Check "ZE '' is a no-op"                  (-not (Test-ZeAffinityMaskHidesAll ""))
Check "ZE whitespace is a no-op"          (-not (Test-ZeAffinityMaskHidesAll "   "))
Check "ZE unset is a no-op"               (-not (Test-ZeAffinityMaskHidesAll $null))
Check "ZE 'default' is a no-op"           (-not (Test-ZeAffinityMaskHidesAll "default"))
Check "ZE ',' alone is a no-op"           (-not (Test-ZeAffinityMaskHidesAll ",,"))
Check "ZE -1 hides"                       (Test-ZeAffinityMaskHidesAll "-1")
Check "ZE ' -1 ' hides"                   (Test-ZeAffinityMaskHidesAll " -1 ")
Check "ZE -1,-2 hides"                    (Test-ZeAffinityMaskHidesAll "-1,-2")
Check "ZE -1.0 hides"                     (Test-ZeAffinityMaskHidesAll "-1.0")
Check "ZE 0 selects"                      (-not (Test-ZeAffinityMaskHidesAll "0"))
Check "ZE 0.1 selects"                    (-not (Test-ZeAffinityMaskHidesAll "0.1"))
Check "ZE 0,1 selects"                    (-not (Test-ZeAffinityMaskHidesAll "0,1"))
Check "ZE -1,0 selects"                   (-not (Test-ZeAffinityMaskHidesAll "-1,0"))
Check "ZE garbage fails open"             (-not (Test-ZeAffinityMaskHidesAll "hello"))
Check "ZE '-' alone fails open"           (-not (Test-ZeAffinityMaskHidesAll "-"))
Check "an unannounced host is not masked" (-not (Test-Arm "" @{ CUDA_VISIBLE_DEVICES = "-1" }))
# Not $TorchIndexPinned / $CuTag: those are $null on the run this check exists for.
Check "the pin is resolved fresh"         ($report -match '\$_gpuCheckPinLeaf = Get-TorchIndexLeaf \(Get-PinnedTorchIndexUrl\)')
Check "a GPU pin is not excluded with it" (-not ($reportCode -match '\$CuTag|\$TorchIndexPinned'))
Check "a missing interpreter is excluded" ($report -match 'Test-Path -LiteralPath \$_gpuCheckPy')
Check "it probes the venv interpreter"    ($report -match '\$_gpuCheckPy = Join-Path \$VenvDir "Scripts\\python\.exe"')
Check "the probe is called once"          ((([regex]::Matches($report, 'Get-TorchGpuVisibility')).Count) -eq 1)

Write-Host "the announced GPU is quoted from the summary, not re-derived"
# The raw flags disagree with the summary on an Intel Arc host that also has an unwheeled AMD card.
Check "it quotes the summary"             ($reportCode -match '\$_gpuCheckAnnounced = \$script:GpuSummaryAnnounced')
Check "it re-derives nothing"             (-not ($reportCode -match '\$_gpuCheckAnnounced = "'))
Check "it reads no detection flag"        (-not ($reportCode -match '\$script:ROCmGfxArch|\$HasROCm|\$HasNvidiaSmi|\$script:IsIntelXpu'))

# ...and the summary itself records what it printed, so the real chain is run, not read.
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

# The reported combination: Intel Arc plus an AMD card on an arch outside $_rocmWheelArches.
$sHybrid = Invoke-Summary @{ IsIntelXpu = $true; IntelGpuLabel = "Intel Arc B580"; ROCmGfxArch = "gfx90c" }
Check "Intel wins the summary"            ($sHybrid.Lines -match 'STEP\|gpu\|Intel GPU detected')
Check "and Intel is what is announced"    ($sHybrid.Announced -eq "Intel GPU")
Check "no AMD announcement is invented"   (-not ($sHybrid.Announced -match 'AMD'))

$sAmd = Invoke-Summary @{ ROCmGfxArch = "gfx1201"; AmdHasGpuWheels = $true }
Check "a ROCm-wheel AMD host is announced" ($sAmd.Announced -eq "AMD GPU (gfx1201)")
$sHip = Invoke-Summary @{ HasROCm = $true; ROCmGpuLabel = "AMD ROCm (gfx1100)"; ROCmGfxArch = "gfx1100"; AmdHasGpuWheels = $true }
Check "a HIP SDK AMD host is announced"   ($sHip.Announced -eq "AMD GPU (gfx1100)")

# Nothing to reconcile: no accelerator, or an AMD card the install path sends to CPU torch by design.
$sNone = Invoke-Summary @{}
Check "a CPU-only host announces nothing" ($null -eq $sNone.Announced)
$sUnknownArch = Invoke-Summary @{ ROCmGpuLabel = "AMD Radeon 780M" }
Check "an arch-less AMD host is not accused" ($null -eq $sUnknownArch.Announced)
# Vega / RDNA1 / MI300 on Windows: the arch is outside $_rocmWheelArches, so torch is CPU-only.
$sVega = Invoke-Summary @{ HasROCm = $true; ROCmGpuLabel = "AMD ROCm (gfx900)"; ROCmGfxArch = "gfx900" }
Check "an unmapped arch on the ROCm arm is not accused" ($null -eq $sVega.Announced)
Check "and the summary still announced it"  ($sVega.Lines -match 'STEP\|gpu\|AMD ROCm \(gfx900\)')
$sVegaSdk = Invoke-Summary @{ HipSdkInstalled = $true; ROCmGpuLabel = "AMD Radeon VII"; ROCmGfxArch = "gfx906" }
Check "an unmapped arch on the HIP SDK arm is not accused" ($null -eq $sVegaSdk.Announced)
$sVegaBundled = Invoke-Summary @{ ROCmGfxArch = "gfx1010" }
Check "an unmapped arch on the bundled-wheel arm is not accused" ($null -eq $sVegaBundled.Announced)

# ...unless a pin overrides it: the pinned path routes a gfx*/rocm* leaf through the GPU index.
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
# The pin predicate is run: `$null -ne "cpu"` is TRUE in PowerShell, so the emptiness half is what
# stops every unpinned host announcing again.
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

# _detect_hardware_locked falls through CUDA -> XPU, so a hybrid host on the XPU wheel answers
# SeesGpu=False and still runs on the GPU (#8473).
Check "the XPU suppression is present"    ($report -match 'SeesXpu')
Check "it suppresses, not merely reports" ($report -match 'not \$_gpuVisibility\.SeesXpu')
Check "it guards the mismatch arm"        ($report -match '\$_gpuVisibility\.SeesGpu -and[\s\S]{0,120}SeesXpu')
Check "the suppression reads the probe, not the wheel label" (-not ($report -match 'Test-VenvTorchIsXpu'))


# --- Test-OneApiSelectorExcludesGpu ------------------------------------------------------------
# Only forms that PROVABLY admit no GPU may suppress; an ordinal can name a GPU. Semantics from
# Intel's SYCL reference: `<backend>:<devices>` terms, `!` discards, discards imply an accept-all.
Write-Host ""
Write-Host "=== ONEAPI_DEVICE_SELECTOR ==="

foreach ($case in @(
    @{ V = "*:cpu";                E = $true;  W = "every backend, cpu only" },
    @{ V = "opencl:cpu";           E = $true;  W = "one backend, cpu only" },
    @{ V = " *:cpu ";              E = $true;  W = "padded" },
    @{ V = "*:CPU";                E = $true;  W = "case folded" },
    @{ V = "opencl:cpu;level_zero:cpu"; E = $true; W = "several accepts, all cpu" },
    @{ V = "!*:gpu";               E = $true;  W = "discard every gpu" },
    @{ V = "!*:*";                 E = $true;  W = "discard everything" },
    @{ V = "*:*;!*:gpu";           E = $true;  W = "accept all then discard gpu" },
    @{ V = "level_zero:*";         E = $false; W = "a backend wildcard admits the gpu" },
    @{ V = "*:gpu";                E = $false; W = "gpu asked for explicitly" },
    @{ V = "*:0";                  E = $false; W = "an ordinal may BE the gpu" },
    @{ V = "*:cpu;level_zero:gpu"; E = $false; W = "one accept still admits a gpu" },
    @{ V = "!level_zero:gpu";      E = $false; W = "one backend discarded, others remain" },
    @{ V = "";                     E = $false; W = "unset" },
    @{ V = "   ";                  E = $false; W = "whitespace only" },
    @{ V = "garbage";              E = $false; W = "unparseable fails open" }
)) {
    Check "$($case.W): '$($case.V)'" ((Test-OneApiSelectorExcludesGpu $case.V) -eq $case.E)
}
Check "a null selector fails open" ((Test-OneApiSelectorExcludesGpu $null) -eq $false)

Check "extraction kept the discard arm" ($oneApiFn -match 'StartsWith\("!"\)')

$maskedText = (Get-Content -Raw $setup) -replace "`r`n", "`n"
# Terminated on \n, like the WMI region guard: without it the pattern matches a CRLF checkout too
# and the companion check below silently retires (see test_amd_venv_repair_loop.ps1).
$maskedPat = '(?s)(\$_gpuCheckMasked = if \(.*?\} else \{ \$false \}\n)'
$masked = if ($maskedText -match $maskedPat) { $Matches[1] } else { "" }
Check "the mask chain was found"        ($masked -ne "")
Check "CRLF is normalised, not tolerated" (-not (($maskedText -replace "`n", "`r`n") -match $maskedPat))
Check "the Intel arm calls the helper"  ($masked -match 'Intel\*[\s\S]{0,900}Test-OneApiSelectorExcludesGpu')
# Both Intel hides, and only in the Intel arm: parseAffinityMask returns early on an empty value,
# so an empty ZE_AFFINITY_MASK hides nothing, but a non-empty all-negative one enables no device.
Check "the Intel arm reads the ZE mask" ($masked -match 'Intel\*[\s\S]{0,900}Test-ZeAffinityMaskHidesAll')
Check "and only the Intel arm does"     ((([regex]::Matches($masked, '\$env:ZE_AFFINITY_MASK')).Count) -eq 1)
# Still off-limits: ROCR is Linux-only, so reading it on Windows would mute a real mismatch.
Check "ROCR is still not read here"     (-not ($masked -match '\$env:ROCR_VISIBLE_DEVICES'))


# --- the ARM64 demotion belongs to this check, not to $AmdHasGpuWheels ------------------------
# The first cut of this folded `(Get-HostMachineArch) -ne "arm64"` straight into
# $AmdHasGpuWheels. That flag has seven other consumers, and the stale-venv check is one of them:
# demoting it there expects "cpu", meets the +rocm wheel the ROCm override installs with no
# host-arch gate of its own, wipes the venv, and setup then exits because the venv is gone. A
# green suite could not see it, because nothing drove an ARM64 host holding a ROCm wheel.
Write-Host ""
Write-Host "the ARM64 demotion is scoped to the diagnostic"
$wheelsPat = '(?ms)^\$AmdHasGpuWheels = \[bool\]\(.*?^\)$'
$wheelsExpr = if ($setupText -match $wheelsPat) { $Matches[0] } else { "" }
Check "the wheels flag was found"       ($wheelsExpr -ne "")
Check "it does not read host arch"      (-not ($wheelsExpr -match 'Get-HostMachineArch'))
Check "the arch list still gates it"    ($wheelsExpr -match '_rocmWheelArches')
# The gate asks the machine itself. A "do wheels reach this host" proxy is false for two
# unrelated reasons, ARM64 and an unwheeled arch, and only the first is excusable: an x64 host on
# an unmapped arch with a GPU pin is announced as AMD, so the inverted form called it ARM64 and,
# once the excuse below reads the wheel, silenced a real mismatch.
$armPat = '(?m)^\$_gpuCheckArm64Amd = .*$'
$armExpr = if ($setupText -match $armPat) { $Matches[0] } else { "" }
Check "the ARM64 candidate flag exists" ($armExpr -ne "")
Check "and it asks the machine itself"  ($armExpr -match 'Get-HostMachineArch\) -eq "arm64"')
Check "not a wheel-reachability proxy"  (-not ($armExpr -match 'AmdWheelsReachThisHost'))
Check "and the proxy is gone entirely"  (-not ($reportCode -match 'AmdWheelsReachThisHost'))
Check "scoped to the AMD arm"           ($reportCode -match '\$_gpuCheckArm64Amd = \(\$_gpuCheckAnnounced -like "AMD\*"\)')

# ...and the ARM64 excuse is decided on the WHEEL, not on the host. Excusing ARM64 outright
# contradicted the very fact that made the demotion necessary: the ROCm override has no host-arch
# gate, so this host can hold a real +rocm wheel, and silencing a broken one is the bug #8473
# exists to report. torch.version.hip is a build constant, so it names the wheel even when the
# runtime is dead.
Check "the gate no longer excuses on host alone" (-not ($reportCode -match '-not \$_gpuCheckArm64Amd -and'))
Check "the excuse reads the wheel"      ($reportCode -match '\$_gpuCheckArm64Excused = \$_gpuCheckArm64Amd -and -not \$_gpuVisibility\.Hip')
Check "and the report honours it"       ($reportCode -match '-not \$_gpuCheckArm64Excused')
# The excuse must be decided after the probe, or it reads a Hip that does not exist yet.
$_excuseLine = ($setupText -split "`n" | Select-String -Pattern '\$_gpuCheckArm64Excused = ' | Select-Object -First 1).LineNumber
$_probeLine  = ($setupText -split "`n" | Select-String -Pattern '\$_gpuVisibility = Get-TorchGpuVisibility' | Select-Object -First 1).LineNumber
Check "it is decided after the probe"   ($_probeLine -and $_excuseLine -and $_probeLine -lt $_excuseLine)

# The venv-wipe chain itself, driven rather than asserted about. An ARM64 host with a wheeled
# arch and an installed +rocm wheel must expect "rocm" and stand pat.
$archesPat = '(?ms)^\$_rocmWheelArches = @\(.*?^\)$'
$archesExpr = if ($setupText -match $archesPat) { $Matches[0] } else { "" }
Check "the arch list was found"         ($archesExpr -match 'gfx1201')
$expectedTagBody = @'
param($HostArch, $InstalledTag, $GfxArch)
function Get-HostMachineArch { return $HostArch }
$script:ROCmGfxArch = $GfxArch
$installedTorchTag = $InstalledTag
__ARCHES__
__WHEELS__
__REACH__
if ($AmdHasGpuWheels) {
    if ($installedTorchTag -eq "cpu") { "cpu" } else { "rocm" }
} else { "cpu" }
'@
# .Replace(), not -replace: the extracted source is full of `$_rocmWheelArches`, and in a regex
# replacement string PowerShell eats the `$_`.
$expectedTagBody = $expectedTagBody.Replace('__ARCHES__', $archesExpr).Replace(
    '__WHEELS__', $wheelsExpr).Replace('__REACH__', $reachExpr)

function Get-ExpectedTag {
    param([string] $HostArch, [string] $InstalledTag, [string] $GfxArch = "gfx1201")
    return [string] (& ([scriptblock]::Create($script:expectedTagBody)) $HostArch $InstalledTag $GfxArch)
}
$script:expectedTagBody = $expectedTagBody
Check "ARM64 keeps an installed rocm venv" ((Get-ExpectedTag "arm64" "rocm") -eq "rocm")
Check "x64 keeps it too"                   ((Get-ExpectedTag "x64" "rocm") -eq "rocm")
Check "a cpu wheel is left alone"          ((Get-ExpectedTag "arm64" "cpu") -eq "cpu")
Check "an unmapped arch expects cpu"       ((Get-ExpectedTag "x64" "rocm" "gfx900") -eq "cpu")


# --- a deliberate local CPU fallback is not a mismatch -----------------------------------------
# $ROCmCpuFallback / $XpuCpuFallback are set when THIS run failed to install the GPU wheel and
# force-installed CPU torch instead. $InstallerTorchTag carries neither, so without these terms
# the user reads the install failure and then a red accusation about the result of it.
Write-Host ""
Write-Host "it does not accuse a run that already explained itself"
Check "a local ROCm fallback is excluded" ($report -match '-not \$_gpuCheckLocalCpuFallback')
Check "it reads both fallback flags"      (($reportCode -match '\$ROCmCpuFallback') -and ($reportCode -match '\$XpuCpuFallback'))
# Both are assigned inside `if (-not $SkipPythonDeps)`. The fast path never enters that block, and
# the fast path is the run this whole check exists for, so a caller's Set-StrictMode would make
# the read fatal on exactly the reported host.
$_fallbackDecl = [regex]::Match($setupText,
    '(?ms)^\$ROCmCpuFallback = \$false\r?\n\$XpuCpuFallback = \$false\r?\n\r?\nif \(-not \$SkipPythonDeps\) \{')
Check "both are declared before the branch" ($_fallbackDecl.Success)
# $ROCmCpuFallback records what setup DECIDED. install_python_stack.py runs after it and its
# Windows _ensure_rocm_torch retries the AMD index precisely because setup fell back, so the venv
# can hold a ROCm wheel by the time this check runs -- and a ROCm wheel that cannot see its GPU is
# the report. The excuse holds only while a CPU wheel is really what is there.
Check "the rocm excuse reads the wheel" ($reportCode -match '\$ROCmCpuFallback -and -not \(Test-VenvTorchIsRocm -VenvPath \$VenvDir\)')
# XPU is NOT reconciled: _ensure_xpu_torch returns on Windows, so nothing reinstalls over that CPU
# fallback, and reading the wheel there would contradict the probe-based XPU suppression above.
Check "the xpu flag stands alone"       ($reportCode -match '\$_gpuCheckLocalCpuFallback = \$XpuCpuFallback -or')
# Short-circuits before the disk read on the fast path, where the flag is never set.
Check "the flag is read first"          ($reportCode -match '\$ROCmCpuFallback -and -not \(Test-Venv')

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
