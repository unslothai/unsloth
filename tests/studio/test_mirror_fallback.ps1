#!/usr/bin/env pwsh
# install.ps1's mirror fallback (synced into setup.ps1): the probe against a local server, and the
# env vars each probe outcome exports. Run: pwsh -NoProfile -File tests/studio/test_mirror_fallback.ps1

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest  # setup.ps1 runs under its caller's strict mode.
$installPath = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1"))).Path
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { throw "install.ps1 has parse errors" }
$setupAst = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1"))).Path, [ref]$tokens, [ref]$errors)
foreach ($pair in @($ast, 'Test-MirrorConfigured'), @($ast, 'Start-MirrorProbe'), @($ast, 'Wait-MirrorProbe'), @($ast, 'Invoke-MirrorFallback'), @($ast, 'Install-UvFromRelease'), @($setupAst, 'Install-UvFromPinnedRelease'), @($setupAst, 'Get-UvInstallDir')) {
    $fn = $pair[0].FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $pair[1] }, $true)
    Invoke-Expression $fn[0].Extent.Text
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" } else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# 206 only for exactly the 1 KiB range on a fresh connection, 403 without a User-Agent (as on CERNET); /slow sends 512 bytes and stalls, /whole ignores Range, /late answers after 2 s.
$server = @'
import http.server, time
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/redirect":
            self.send_response(302); self.send_header("Location", "/ok"); self.send_header("Content-Length", "0"); self.end_headers(); return
        if self.path == "/late": time.sleep(2)
        ranged = self.headers.get("Range") == "bytes=0-1023" and self.headers.get("Connection", "").lower() == "close" and self.path != "/whole"
        code = 404 if self.path == "/404" else 403 if not self.headers.get("User-Agent") else 206 if ranged else 200; size = 1024 if ranged else 1 << 20; first = 512 if self.path == "/slow" else 1
        self.send_response(code); self.send_header("Content-Length", str(size)); self.end_headers()
        self.wfile.write(b"x" * first); self.wfile.flush()
        time.sleep(5 if self.path == "/slow" else 0); self.wfile.write(b"x" * (size - first))
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer.request_queue_size = 64; s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); print(s.server_port, flush=True); s.serve_forever()
'@
# Windows' python3 is often the Microsoft Store stub, which prints an install hint instead of running.
$python = if ($env:OS -ne 'Windows_NT' -and (Get-Command python3 -ErrorAction SilentlyContinue)) { 'python3' } else { 'python' }
$psi = New-Object System.Diagnostics.ProcessStartInfo $python, "-c `"exec(__import__('sys').stdin.read())`""
$psi.RedirectStandardInput = $true; $psi.RedirectStandardOutput = $true; $psi.UseShellExecute = $false
$proc = [System.Diagnostics.Process]::Start($psi)
$proc.StandardInput.Write($server); $proc.StandardInput.Close()
$base = "http://127.0.0.1:$($proc.StandardOutput.ReadLine())"
try {
    [System.Net.ServicePointManager]::DefaultConnectionLimit = 16  # as Invoke-MirrorFallback does: the batch below opens 6 connections to one host.
    $late = Start-MirrorProbe -Urls @("$base/late") -Seconds 1 -LastByte 1023; Start-Sleep -Seconds 3; Check "probe: an answer after the deadline stays late, however late it is collected" ((Wait-MirrorProbe $late)["$base/late"][0] -eq 0)
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Wait-MirrorProbe ($state = Start-MirrorProbe -Urls @("$base/ok", "$base/slow", "$base/whole", "$base/late", "$base/404", "http://127.0.0.1:9/") -Seconds 1.5 -LastByte 1023)
    Check "probe: gives up at its deadline, cancelling what has not answered" ($clock.Elapsed.TotalSeconds -lt 3 -and $r["$base/late"][0] -eq 0 -and [System.Threading.Tasks.Task]::WaitAny(@($state.Heads["$base/late"]), 200) -eq 0)
    Check "probe: a ranged 1 KiB answer is 206 with its speed" ($r["$base/ok"][0] -eq 206 -and $r["$base/ok"][1] -gt 1024)
    Check "probe: redirects are followed" ((Wait-MirrorProbe (Start-MirrorProbe -Urls @("$base/redirect") -Seconds 1.5 -LastByte 1023))["$base/redirect"][0] -eq 206)  # alone: pwsh 7 gets resets from this server on side-by-side redirects (real mirrors measured clean)
    Check "probe: a stalled body reports the speed so far" ($r["$base/slow"][0] -eq 206 -and $r["$base/slow"][1] -gt 200 -and $r["$base/slow"][1] -lt 600)
    Check "probe: an answer that ignores Range is capped, not buffered whole" ($r["$base/whole"][0] -eq 200 -and $state.Buffers["$base/whole"].Position -in 1024..(1024 + 65536))
    Check "probe: an HTTP error keeps its code" ($r["$base/404"][0] -eq 404 -and $r["$base/404"][1] -eq 0)
    Check "probe: a refused connection is code 0" ($r["http://127.0.0.1:9/"][0] -eq 0)
} finally { $proc.Kill() }

$names = '_UNSLOTH_MIRROR_PROBED', 'UNSLOTH_MIRROR_FALLBACK', 'UV_INDEX', 'UV_DEFAULT_INDEX', 'UV_INDEX_URL', 'UV_INDEX_STRATEGY', 'PIP_INDEX_URL',
    'PIP_EXTRA_INDEX_URL', 'UNSLOTH_PYTORCH_MIRROR', 'UNSLOTH_NODE_MIRROR', 'UNSLOTH_NPM_REGISTRY', 'UV_CONFIG_FILE', 'PIP_CONFIG_FILE',
    'UNSLOTH_UV_WHEEL_MIRROR', 'UV_INSTALLER_GITHUB_BASE_URL', 'UV_INSTALL_DIR', 'UV_NO_MODIFY_PATH'
$saved = @{}; foreach ($n in $names + 'APPDATA', 'USERPROFILE', 'ProgramData', 'PATH') { $saved[$n] = [Environment]::GetEnvironmentVariable($n) }
$home_ = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-mirror-" + [guid]::NewGuid())
$pipIni = Join-Path $home_ 'AppData/pip/pip.ini'
New-Item -ItemType Directory -Force -Path (Split-Path $pipIni) | Out-Null
$env:APPDATA = Join-Path $home_ 'AppData'; $env:USERPROFILE = $home_; $env:ProgramData = Join-Path $home_ 'ProgramData'
function step($l, $v) { $script:lines += "STEP $v" }
function substep($m) { $script:lines += "SUBSTEP $m" }
$M = 'https://mirrors.cernet.edu.cn'
# Every probe answers from $script:mock by class: fast, slow (<1 MiB/s), blocked or "<code> <bytes/s>"; "a|b" is a, then b.
function Get-ProbeClass([string]$u) {
    switch -Wildcard ($u) {
        "$M/pypi/web/simple/*" { return 'cernetpypiindex' } "$M/pytorch/whl/cpu/torch/" { return 'cernettorchindex' } "$M/pytorch/*" { return 'cernettorch' }
        "$M/nodejs-release/*" { return 'cernetnode' } "$M/*" { return 'cernet' } 'https://registry.npmmirror.com/*' { return 'npmmirror' }
        'https://pypi.org/*' { return 'pypiindex' } 'https://download.pytorch.org/*' { return 'torchindex' } 'https://files.pythonhosted.org/*' { return 'pypi' }
        'https://download-r2.pytorch.org/*' { return 'torch' } 'https://nodejs.org/*' { return 'node' } 'https://registry.npmjs.org/*' { return 'npm' } default { return 'astral' }
    }
}
function Start-MirrorProbe([string[]]$Urls, [double]$Seconds, [long]$LastByte) {
    if ($Seconds -eq 1.5) { if ($script:timing -or $Urls.Count -gt 1) { $script:overlap = $true }; $script:timing = $true }
    $script:probed += $Urls; $script:protocols += "$([System.Net.ServicePointManager]::SecurityProtocol)/$([System.Net.ServicePointManager]::DefaultConnectionLimit)"; foreach ($u in $Urls) { if (($u -match '/(simple/uv|whl/cpu/torch)/$') -ne ($LastByte -eq 1023) -or $Seconds -notin $(if ($LastByte -eq 1023) { 4 } else { 1.5, 4 })) { $script:badRange = $true } }
    return @{ Urls = $Urls; Seconds = $Seconds }
}
function Wait-MirrorProbe($Probe) {
    if ($Probe.Seconds -eq 1.5) { $script:timing = $false }
    $out = @{}
    foreach ($u in $Probe.Urls) {
        $c = Get-ProbeClass $u; $r = "$($script:mock[$c])"; if (-not $r) { $r = 'fast' }
        $r = if ($script:seen -contains $c) { $r.Split('|')[-1] } else { $script:seen += $c; $r.Split('|')[0] }
        $out[$u] = switch ($r) { 'fast' { @(206, [long]4000000) } 'slow' { @(206, [long]300000) } 'blocked' { @(0, [long]0) } default { @([int]$r.Split(' ')[0], [long]$r.Split(' ')[1]) } }
    }
    return $out
}
function Run($mock, $envs = @{}) {
    foreach ($n in $names) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
    foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
    $script:mock = $mock; $script:seen = @(); $script:lines = @(); $script:probed = @(); $script:protocols = @(); $script:badRange = $false; $script:timing = $false; $script:overlap = $false
    Invoke-MirrorFallback
}
$pypiMirror = "$M/pypi/web/simple"
try {
    Run @{}; Check "fast hosts export and print nothing" (-not $env:UV_DEFAULT_INDEX -and -not $env:UNSLOTH_NPM_REGISTRY -and $script:lines.Count -eq 0)
    Check "fast hosts never touch a mirror" (-not ($script:probed -match 'cernet|npmmirror'))
    Run @{ pypi = 'slow'; torch = 'slow'; node = 'slow'; npm = 'slow'; astral = 'slow' }; Check "defaults are timed one at a time, on 1 MiB within 1.5 s or 4 s; indexes on 1 KiB within 4 s" (-not $script:overlap -and -not $script:badRange)
    Run @{ pypi = 'blocked' }
    Check "blocked pypi: mirror is the only uv/pip index" ($env:UV_DEFAULT_INDEX -eq $pypiMirror -and $env:PIP_INDEX_URL -eq $pypiMirror -and -not $env:UV_INDEX -and -not $env:UV_INDEX_STRATEGY -and -not $env:PIP_EXTRA_INDEX_URL)
    Check "blocked pypi: says so with both speeds and names the opt-out" ($script:lines -contains "STEP PyPI is blocked (0 KB/s, mirror 3906 KB/s); using $pypiMirror" -and $script:lines[-1] -like 'SUBSTEP Set UNSLOTH_MIRROR_FALLBACK=0*')
    Run @{ pypi = 'slow' }
    Check "slow pypi: mirror first, pypi.org second, unsafe-first-match" ($env:UV_INDEX -eq $pypiMirror -and $env:UV_DEFAULT_INDEX -eq 'https://pypi.org/simple' -and $env:UV_INDEX_STRATEGY -eq 'unsafe-first-match' -and $env:PIP_INDEX_URL -eq $pypiMirror -and $env:PIP_EXTRA_INDEX_URL -eq 'https://pypi.org/simple')
    Check "slow pypi: the default is timed again in the race" (@($script:probed -like 'https://files.pythonhosted.org/*').Count -eq 2)
    Run @{ pypi = 'slow' } @{ UV_INDEX_STRATEGY = 'first-index' }; Check "a user index strategy is kept" ($env:UV_INDEX_STRATEGY -eq 'first-index')
    Run @{ pypi = 'slow'; cernet = '206 200000' }; Check "a mirror slower than the default is not used" (-not $env:UV_INDEX -and $script:lines.Count -eq 0)
    Run @{ pypi = 'blocked'; cernet = 'blocked' }; Check "a dead mirror changes and prints nothing" (-not $env:UV_DEFAULT_INDEX -and $script:lines.Count -eq 0)
    Run @{ pypi = 'slow|206 2000000'; cernet = '206 3000000' }; Check "a default back above 1 MiB/s in the race is kept" (-not $env:UV_INDEX -and $script:lines.Count -eq 0)
    Run @{ pypi = '302 0' }; Check "a redirect that stalls counts as blocked" ($env:UV_DEFAULT_INDEX -eq $pypiMirror)
    Run @{ pypi = 'slow|404 5000000'; cernet = '206 400000' }; Check "an HTTP error in the race weighs as 0 B/s" ($script:lines -contains "STEP PyPI is blocked (0 KB/s, mirror 390 KB/s); using $pypiMirror")
    Run @{ pypi = '404 0' }; Check "a default answering an HTTP error is kept, without timing the mirror" (-not $env:UV_DEFAULT_INDEX -and -not ($script:probed -like "$M/*"))
    Run @{ torch = 'blocked'; node = 'slow'; npm = 'blocked' }
    Check "torch, node and npm mirrors; healthy pypi untouched" ($env:UNSLOTH_PYTORCH_MIRROR -eq "$M/pytorch/whl" -and $env:UNSLOTH_NODE_MIRROR -eq "$M/nodejs-release" -and $env:UNSLOTH_NPM_REGISTRY -eq 'https://registry.npmmirror.com' -and -not $env:PIP_INDEX_URL)
    Check "each CERNET tree is timed once" (@($script:probed -like "$M/*" -match '\.(gz|whl)$').Count -eq 2)
    Run @{ torch = 'blocked'; node = 'blocked'; cernetnode = 'blocked' }; Check "torch and node are weighed against their own CERNET trees" ($env:UNSLOTH_PYTORCH_MIRROR -eq "$M/pytorch/whl" -and -not $env:UNSLOTH_NODE_MIRROR)
    Run @{ pypiindex = 'blocked' }; Check "an unreachable pypi.org means blocked mode" ($env:UV_DEFAULT_INDEX -eq $pypiMirror -and -not $env:UV_INDEX)
    Run @{ torchindex = 'blocked' }; Check "an unreachable torch index switches torch" ($env:UNSLOTH_PYTORCH_MIRROR -eq "$M/pytorch/whl")
    Run @{ pypi = 'blocked'; torch = 'blocked'; astral = 'blocked'; cernetpypiindex = 'blocked' }
    Check "a mirror is used only while its own index answers; the uv wheel skips the index" (-not $env:PIP_INDEX_URL -and $env:UNSLOTH_PYTORCH_MIRROR -and $env:UNSLOTH_UV_WHEEL_MIRROR -eq "$M/pypi/web")
    Run @{ torch = 'blocked'; cernettorchindex = 'blocked' }; Check "a dead CERNET torch index keeps torch" (-not $env:UNSLOTH_PYTORCH_MIRROR)
    $limit = [System.Net.ServicePointManager]::DefaultConnectionLimit; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls; [System.Net.ServicePointManager]::DefaultConnectionLimit = 2
    try { Run @{ pypi = 'blocked' } } finally { $protocol = "$([System.Net.ServicePointManager]::SecurityProtocol)/$([System.Net.ServicePointManager]::DefaultConnectionLimit)"; [System.Net.ServicePointManager]::SecurityProtocol = 0; [System.Net.ServicePointManager]::DefaultConnectionLimit = $limit }
    Check "legacy TLS and 2 connections per host are raised for the probes and restored" ($protocol -eq 'Tls/2' -and -not ($script:protocols -ne 'Tls12/16'))
    Run @{ pypi = 'blocked' } @{ UV_INDEX_URL = 'https://corp.example/simple' }; Check "user uv index env: only pip falls back" (-not $env:UV_DEFAULT_INDEX -and $env:PIP_INDEX_URL)
    New-Item -ItemType Directory -Force -Path (Join-Path $env:APPDATA 'uv') | Out-Null
    foreach ($toml in "[[index]]`nurl = 'https://corp.example/simple'", "index = [{ url = 'https://corp.example/simple', default = true }]", "pip.index-url = 'https://corp.example/simple'") {
        Set-Content -Path (Join-Path $env:APPDATA 'uv/uv.toml') -Value $toml
        Run @{ pypi = 'blocked' }; Check "uv.toml $($toml.Split("`n")[0]): only pip falls back" (-not $env:UV_DEFAULT_INDEX -and $env:PIP_INDEX_URL)
    }
    Remove-Item -LiteralPath (Join-Path $env:APPDATA 'uv/uv.toml')
    Set-Content -Path $pipIni -Value "[global]`nindex-url = https://corp.example/simple"
    Run @{ pypi = 'blocked' }; Check "pip.ini index: only uv falls back" (-not $env:PIP_INDEX_URL -and $env:UV_DEFAULT_INDEX)
    Set-Content -Path $pipIni -Value "[global]`ntimeout = 60"
    Run @{ pypi = 'blocked' }; Check "pip.ini without an index still falls back" ($env:PIP_INDEX_URL -eq $pypiMirror)
    Run @{ torch = 'blocked' } @{ UNSLOTH_TORCH_INDEX_URL = 'https://corp.example/whl/cu128' }; Check "a pinned torch index is not overridden" (-not $env:UNSLOTH_PYTORCH_MIRROR)
    Run @{ npm = 'blocked' } @{ UNSLOTH_NPM_REGISTRY = 'https://corp.example/npm/' }; Check "a user npm registry is kept and not probed" ($env:UNSLOTH_NPM_REGISTRY -eq 'https://corp.example/npm/' -and -not ($script:probed -like '*npm*'))
    Run @{ pypi = 'blocked' } @{ UNSLOTH_MIRROR_FALLBACK = 'Off' }; Check "UNSLOTH_MIRROR_FALLBACK=Off probes nothing" ($script:probed.Count -eq 0)
    Run @{ pypi = 'blocked' } @{ _UNSLOTH_MIRROR_PROBED = '1' }; Check "a parent installer's probe is not repeated" ($script:probed.Count -eq 0)
    Run @{ astral = 'blocked' }; Check "blocked uv releases: the pinned wheel comes from the PyPI mirror" ($env:UNSLOTH_UV_WHEEL_MIRROR -eq "$M/pypi/web"); foreach ($src in 'UV_INSTALLER_GITHUB_BASE_URL', 'UNSLOTH_UV_WHEEL_MIRROR') { Run @{ astral = 'blocked' } @{ $src = 'https://corp.example/uv' }; Check "a user $src skips the uv probe" (-not ($script:probed -like '*releases.astral.sh*')) }
    # Both uv installers against a fake wheel: the mirror is the only source and the wheel digest gates the install.
    function Get-HostMachineArch { 'x86_64' }; function Get-UvHostArch { 'x86_64' }; function Get-UvExecutableVerdict { 'ok' }; function Get-SetupUvExecutableVerdict { 'ok' }
    function Invoke-WebRequest([switch]$UseBasicParsing, $OutFile, $Uri) { $script:fetched += $Uri; Copy-Item "$home_/uv.zip" $OutFile }
    foreach ($exe in 'uv.exe', 'uvx.exe') { New-Item -Force -Path "$home_/whl/uv-0.12.1.data/scripts/$exe" -Value $exe | Out-Null }
    Compress-Archive -Path "$home_/whl/uv-0.12.1.data" -DestinationPath "$home_/uv.zip"
    $UvPinnedVersion = '0.12.1'; $wheelSha = (Get-FileHash "$home_/uv.zip").Hash
    foreach ($fn in 'Install-UvFromRelease', 'Install-UvFromPinnedRelease') { foreach ($good in $true, $false) {
        $dest = "$home_/bin-$fn-$good"; $env:UV_INSTALL_DIR = $dest; $env:UV_NO_MODIFY_PATH = '1'; $env:UNSLOTH_UV_WHEEL_MIRROR = 'https://m.example/pypi/web/'; $script:fetched = @()
        $UvPinnedAssets = @{ x86_64 = @{ Asset = 'uv-x86_64-pc-windows-msvc.zip'; Sha256 = 'X'; Wheel = 'packages/ab/uv.whl'; WheelSha256 = $(if ($good) { $wheelSha } else { '0' * 64 }) } }
        $installed = (@(& $fn)[-1] -eq $true) -and (Test-Path "$dest/uv.exe") -and (Test-Path "$dest/uvx.exe") -and "$script:fetched" -eq 'https://m.example/pypi/web/packages/ab/uv.whl'
        Check "${fn}: wheel mirror, good digest $good" ($installed -eq $good -and ($good -or -not (Test-Path "$dest/uv.exe")))
    } }
} finally {
    foreach ($n in $saved.Keys) { if ($null -eq $saved[$n]) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue } else { Set-Item "Env:$n" $saved[$n] } }
    Remove-Item -LiteralPath $home_ -Recurse -Force -ErrorAction SilentlyContinue
}
if ($failures) { throw "$failures mirror fallback check(s) failed" }
