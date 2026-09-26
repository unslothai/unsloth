#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest  # setup.ps1 runs under its caller's strict mode.
$installPath = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1"))).Path
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { throw "install.ps1 has parse errors" }
$setupAst = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1"))).Path, [ref]$tokens, [ref]$errors)
$load = 'Test-UvEnvFlag', 'Test-MirrorConfigured', 'Start-MirrorProbe', 'Wait-MirrorProbe', 'Invoke-MirrorFallback', 'Get-MirrorName', 'Set-MirrorEnv', 'Pop-MirrorSpare', 'Use-MirrorSpare', 'Get-MirrorFailedHost',
    'Install-UvFromRelease', 'Redact-InstallOutput', 'Invoke-InstallCommand', 'Invoke-InstallMirrorRetry', 'Invoke-InstallCommandRetry'
foreach ($pair in @($load | ForEach-Object { , @($ast, $_) }) + (@('Install-UvFromPinnedRelease', 'Get-UvInstallDir', 'Invoke-SetupCommand', 'Invoke-NpmMirrorRetry') | ForEach-Object { , @($setupAst, $_) })) {
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
    'PIP_EXTRA_INDEX_URL', 'UNSLOTH_PYTORCH_MIRROR', 'UNSLOTH_NODE_MIRROR', 'UNSLOTH_NPM_REGISTRY', 'NPM_CONFIG_REGISTRY', 'UV_OFFLINE', 'UV_CONFIG_FILE', 'PIP_CONFIG_FILE',
    'UNSLOTH_UV_WHEEL_MIRROR', 'UV_INSTALLER_GITHUB_BASE_URL', 'UV_INSTALL_DIR', 'UV_NO_MODIFY_PATH', '_UNSLOTH_MIRROR_SPARE', 'UNSLOTH_INSTALL_RETRIES', 'UNSLOTH_INSTALL_RETRY_DELAY'
$saved = @{}; foreach ($n in $names + 'APPDATA', 'USERPROFILE', 'ProgramData', 'PATH') { $saved[$n] = [Environment]::GetEnvironmentVariable($n) }
$home_ = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-mirror-" + [guid]::NewGuid())
$pipIni = Join-Path $home_ 'AppData/pip/pip.ini'
New-Item -ItemType Directory -Force -Path (Split-Path $pipIni) | Out-Null
$env:APPDATA = Join-Path $home_ 'AppData'; $env:USERPROFILE = $home_; $env:ProgramData = Join-Path $home_ 'ProgramData'
function step($l, $v, $c) { $script:lines += "STEP $v" }
function substep($m) { $script:lines += "SUBSTEP $m" }
$M = 'https://tuna.mirrors.cernet.edu.cn'
function Get-ProbeClass([string]$u) {
    switch -Wildcard ($u) {
        "$M/pypi/web/simple/*" { return 'cernetpypiindex' } "$M/pytorch/whl/cpu/torch/" { return 'cernettorchindex' } "$M/pytorch/*" { return 'cernettorch' }
        'https://registry.npmmirror.com/-/binary/node/*' { return 'npmmirrornode' } "$M/*" { return 'cernet' } 'https://registry.npmmirror.com/*' { return 'npmmirror' }
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
$real = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Test-MirrorInChina' }, $true)[0].Extent.Text
Invoke-Expression $real
function Get-TimeZone { [pscustomobject]@{ Id = $script:tz } }
$realDns = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Get-MirrorDnsServers' }, $true)[0].Extent.Text
Invoke-Expression $realDns
Check "location: the .NET resolver list reads without throwing" ($null -eq $(try { @(Get-MirrorDnsServers) | Out-Null; $null } catch { $_ }))
function Get-MirrorDnsServers { $script:dns }
foreach ($case in @(@('China Standard Time', $null, $true), @('Asia/Shanghai', $null, $true), @('Taipei Standard Time', $null, $false), @('UTC', $null, $false),
        @('UTC', @('8.8.8.8', '100.100.2.136'), $true), @('UTC', @('223.5.5.5'), $true), @('UTC', @('114.114.115.119'), $true), @('UTC', @('182.254.116.116'), $true), @('UTC', @('100.100.100.100', '1.1.1.1'), $false))) {
    $script:tz = $case[0]; $script:dns = $case[1]
    Check "location: time zone $($case[0]), resolvers $($case[1] -join ',') -> China=$($case[2])" ((Test-MirrorInChina) -eq $case[2])
}
Remove-Item Function:Get-TimeZone, Function:Get-MirrorDnsServers
function Test-MirrorInChina { $script:inChina }

function Run($mock, $envs = @{}, [switch]$SpareOnly, [switch]$Abroad) {
    foreach ($n in $names) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
    foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
    $script:inChina = -not $Abroad
    $script:mock = $mock; $script:seen = @(); $script:lines = @(); $script:probed = @(); $script:protocols = @(); $script:badRange = $false; $script:timing = $false; $script:overlap = $false
    Invoke-MirrorFallback -SpareOnly:$SpareOnly
}
$pypiMirror = "$M/pypi/web/simple"
try {
    Run @{}; Check "fast hosts export and print nothing" (-not $env:UV_DEFAULT_INDEX -and -not $env:UNSLOTH_NPM_REGISTRY -and $script:lines.Count -eq 0)
    Run @{ pypi = 'blocked'; torch = 'blocked'; npm = 'blocked' } -Abroad
    Check "outside China: nothing probed, switched, armed or exported" ($script:probed.Count -eq 0 -and $script:lines.Count -eq 0 -and -not $env:UV_DEFAULT_INDEX -and -not $env:_UNSLOTH_MIRROR_SPARE -and -not $env:_UNSLOTH_MIRROR_PROBED)
    Run @{} -Abroad -SpareOnly; Check "outside China: no retry is armed either" (-not $env:_UNSLOTH_MIRROR_SPARE)
    foreach ($on in '1', 'true', ' Yes ', 'on') { Run @{ pypi = 'blocked' } @{ UV_OFFLINE = $on }; Check "UV_OFFLINE=$($on): nothing probed, retries still armed" ($script:probed.Count -eq 0 -and -not $env:UV_DEFAULT_INDEX -and $env:_UNSLOTH_MIRROR_SPARE -like 'pypi|*') }
    Run @{ pypi = 'blocked' } @{ UV_OFFLINE = '0' }; Check "UV_OFFLINE=0 still probes" ($env:UV_DEFAULT_INDEX -eq "$M/pypi/web/simple")
    Run @{} @{ _UNSLOTH_MIRROR_SPARE = 'pypi|UV_DEFAULT_INDEX=https://x' } -Abroad -SpareOnly; Check "outside China: a retry state inherited from a parent is dropped" (-not $env:_UNSLOTH_MIRROR_SPARE)
    Run @{} @{ _UNSLOTH_MIRROR_SPARE = 'pypi|UV_DEFAULT_INDEX=https://x'; UNSLOTH_MIRROR_FALLBACK = '0' } -SpareOnly; Check "UNSLOTH_MIRROR_FALLBACK=0: a retry state inherited from a parent is dropped" (-not $env:_UNSLOTH_MIRROR_SPARE)
    Run @{ pypi = 'blocked' } @{ UNSLOTH_MIRROR_FALLBACK = '1' } -Abroad; Check "outside China, UNSLOTH_MIRROR_FALLBACK=1 opts in" ($env:UV_DEFAULT_INDEX -eq "$M/pypi/web/simple")
    Run @{ pypi = 'blocked' } @{ UNSLOTH_MIRROR_FALLBACK = '0' }; Check "in China, UNSLOTH_MIRROR_FALLBACK=0 still opts out" ($script:probed.Count -eq 0 -and -not $env:UV_DEFAULT_INDEX)
    Check "fast hosts never touch a mirror" (-not ($script:probed -match 'cernet|npmmirror'))
    Run @{ pypi = 'slow'; torch = 'slow'; node = 'slow'; npm = 'slow'; astral = 'slow' }; Check "defaults are timed one at a time, on 1 MiB within 1.5 s or 4 s; indexes on 1 KiB within 4 s" (-not $script:overlap -and -not $script:badRange)
    Run @{ torch = 'blocked' }
    Check "hosts left on their defaults keep their mirror, blocked-mode, as a spare" ($env:_UNSLOTH_MIRROR_SPARE -eq "pypi|UV_DEFAULT_INDEX=$pypiMirror|PIP_INDEX_URL=$pypiMirror node|UNSLOTH_NODE_MIRROR=https://registry.npmmirror.com/-/binary/node npm|UNSLOTH_NPM_REGISTRY=https://registry.npmmirror.com uvbin|UNSLOTH_UV_WHEEL_MIRROR=$M/pypi/web")
    Run @{ pypi = 'blocked' }
    Check "blocked pypi: mirror is the only uv/pip index" ($env:UV_DEFAULT_INDEX -eq $pypiMirror -and $env:PIP_INDEX_URL -eq $pypiMirror -and -not $env:UV_INDEX -and -not $env:UV_INDEX_STRATEGY -and -not $env:PIP_EXTRA_INDEX_URL)
    Check "blocked pypi: says so with both speeds and names the opt-out" ($script:lines -contains "STEP PyPI is blocked (0 KB/s, mirror 3906 KB/s); using $pypiMirror" -and $script:lines[-1] -like 'SUBSTEP Set UNSLOTH_MIRROR_FALLBACK=0*')
    Run @{ pypi = 'slow' }
    Check "slow pypi: the mirror is the only uv/pip index" ($env:UV_DEFAULT_INDEX -eq $pypiMirror -and $env:PIP_INDEX_URL -eq $pypiMirror -and -not $env:UV_INDEX -and -not $env:UV_INDEX_STRATEGY -and -not $env:PIP_EXTRA_INDEX_URL)
    Check "slow pypi: pypi.org, which still answers, is spared behind the mirror for what it has not synced" ($env:_UNSLOTH_MIRROR_SPARE -like "* unsynced|UV_DEFAULT_INDEX=https://pypi.org/simple|UV_INDEX=$pypiMirror|UV_INDEX_STRATEGY=unsafe-first-match|PIP_EXTRA_INDEX_URL=https://pypi.org/simple|PIP_INDEX_URL=$pypiMirror")
    Check "slow pypi: the default is timed again in the race" (@($script:probed -like 'https://files.pythonhosted.org/*').Count -eq 2)
    Run @{ pypi = 'slow' } @{ UV_INDEX_STRATEGY = 'first-index' }; Check "a user index strategy is kept" ($env:_UNSLOTH_MIRROR_SPARE -like '*|UV_INDEX_STRATEGY=first-index|*')
    Run @{ pypi = 'slow'; cernet = '206 200000' }; Check "a mirror slower than the default is not used" (-not $env:UV_DEFAULT_INDEX -and $script:lines.Count -eq 0)
    Run @{ pypi = 'blocked'; cernet = 'blocked' }; Check "a dead mirror changes and prints nothing" (-not $env:UV_DEFAULT_INDEX -and $script:lines.Count -eq 0)
    Run @{ pypi = 'slow|206 2000000'; cernet = '206 3000000' }; Check "a default back above 1 MiB/s in the race is kept" (-not $env:UV_DEFAULT_INDEX -and $script:lines.Count -eq 0)
    Run @{ pypi = '302 0' }; Check "a redirect that stalls counts as blocked" ($env:UV_DEFAULT_INDEX -eq $pypiMirror)
    Run @{ pypi = 'slow|404 5000000'; cernet = '206 400000' }; Check "an HTTP error in the race weighs as 0 B/s" ($script:lines -contains "STEP PyPI is blocked (0 KB/s, mirror 390 KB/s); using $pypiMirror")
    Run @{ pypi = '404 0' }; Check "a default answering an HTTP error is kept, without timing the mirror" (-not $env:UV_DEFAULT_INDEX -and -not ($script:probed -like "$M/*"))
    Run @{ torch = 'blocked'; node = 'slow'; npm = 'blocked' }
    Check "torch, node and npm mirrors; healthy pypi untouched" ($env:UNSLOTH_PYTORCH_MIRROR -eq "$M/pytorch/whl" -and $env:UNSLOTH_NODE_MIRROR -eq 'https://registry.npmmirror.com/-/binary/node' -and $env:UNSLOTH_NPM_REGISTRY -eq 'https://registry.npmmirror.com' -and -not $env:PIP_INDEX_URL)
    Check "each mirror tree is timed once" (@($script:probed -match "^($([regex]::Escape($M))|https://registry\.npmmirror\.com)/.*\.(t?gz|whl)$").Count -eq 3)
    Run @{ torch = 'blocked'; node = 'blocked'; npmmirrornode = 'blocked' }; Check "torch and node are weighed against their own mirror trees" ($env:UNSLOTH_PYTORCH_MIRROR -eq "$M/pytorch/whl" -and -not $env:UNSLOTH_NODE_MIRROR)
    Run @{ pypiindex = 'blocked' }; Check "an unreachable pypi.org means blocked mode, with nothing spared behind the mirror" ($env:UV_DEFAULT_INDEX -eq $pypiMirror -and $env:_UNSLOTH_MIRROR_SPARE -notmatch 'unsynced')
    Run @{ pypiindex = 'blocked|fast' }; Check "a pypi.org index that answers in the race is not blocked" (-not $env:UV_DEFAULT_INDEX -and $script:lines.Count -eq 0)
    Run @{ npm = 'blocked' } @{ NPM_CONFIG_REGISTRY = 'https://corp.example/npm/' }; Check "NPM_CONFIG_REGISTRY: no npm switch or retry" (-not $env:UNSLOTH_NPM_REGISTRY -and $env:_UNSLOTH_MIRROR_SPARE -notmatch 'UNSLOTH_NPM_REGISTRY')
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
    $proj = Join-Path $home_ 'proj'; New-Item -ItemType Directory -Force -Path (Join-Path $proj 'src') | Out-Null
    Set-Content -Path (Join-Path $proj 'pyproject.toml') -Value "[project]`nname = 'p'`n`n[[tool.uv.index]]`nurl = 'https://corp.example/simple'`ndefault = true"
    Push-Location (Join-Path $proj 'src')
    try {
        Run @{ pypi = 'blocked' }; Check "a parent pyproject.toml [tool.uv] index: only pip falls back" (-not $env:UV_DEFAULT_INDEX -and $env:PIP_INDEX_URL)
        Set-Content -Path (Join-Path $proj 'src/uv.toml') -Value 'native-tls = true'
        Run @{ pypi = 'blocked' }; Check "the nearest uv.toml hides a parent pyproject.toml" ($env:UV_DEFAULT_INDEX -eq $pypiMirror)
    } finally { Pop-Location; Remove-Item -LiteralPath $proj -Recurse -Force }
    Set-Content -Path $pipIni -Value "[global]`nindex-url = https://corp.example/simple"
    Run @{ pypi = 'blocked' }; Check "pip.ini index: only uv falls back" (-not $env:PIP_INDEX_URL -and $env:UV_DEFAULT_INDEX)
    Set-Content -Path $pipIni -Value "[global]`ntimeout = 60"
    Run @{ pypi = 'blocked' }; Check "pip.ini without an index still falls back" ($env:PIP_INDEX_URL -eq $pypiMirror)
    $VenvDir = Join-Path $home_ 'venv'; New-Item -ItemType Directory -Force -Path $VenvDir | Out-Null
    Set-Content -Path (Join-Path $VenvDir 'pip.ini') -Value "[global]`nindex-url = https://corp.example/simple"
    Run @{ pypi = 'blocked' }; Check "the Studio venv's pip.ini index: only uv falls back" (-not $env:PIP_INDEX_URL -and $env:UV_DEFAULT_INDEX)
    Remove-Variable VenvDir
    Run @{ node = 'blocked' } -SpareOnly
    Check "spare only: no host is probed, and every host gets its retry" ($script:probed.Count -eq 0 -and -not $env:_UNSLOTH_MIRROR_PROBED -and -not $env:UNSLOTH_NODE_MIRROR -and $env:_UNSLOTH_MIRROR_SPARE -like "pypi|UV_DEFAULT_INDEX=$pypiMirror|PIP_INDEX_URL=$pypiMirror torch|*node|UNSLOTH_NODE_MIRROR=https://registry.npmmirror.com/-/binary/node *")
    Invoke-MirrorFallback; Check "spare only: the probe still runs later" ($env:UNSLOTH_NODE_MIRROR -eq 'https://registry.npmmirror.com/-/binary/node')
    Run @{ torch = 'blocked' } @{ UNSLOTH_TORCH_INDEX_URL = 'https://corp.example/whl/cu128' }; Check "a pinned torch index is not overridden" (-not $env:UNSLOTH_PYTORCH_MIRROR)
    Run @{ npm = 'blocked' } @{ UNSLOTH_NPM_REGISTRY = 'https://corp.example/npm/' }; Check "a user npm registry is kept and not probed" ($env:UNSLOTH_NPM_REGISTRY -eq 'https://corp.example/npm/' -and -not ($script:probed -like '*npm*'))
    Run @{ pypi = 'blocked' } @{ UNSLOTH_MIRROR_FALLBACK = 'Off' }; Check "UNSLOTH_MIRROR_FALLBACK=Off probes nothing" ($script:probed.Count -eq 0)
    Run @{ pypi = 'blocked' } @{ _UNSLOTH_MIRROR_PROBED = '1' }; Check "a parent installer's probe is not repeated" ($script:probed.Count -eq 0)
    Run @{ astral = 'blocked' }; Check "blocked uv releases: the pinned wheel comes from the PyPI mirror" ($env:UNSLOTH_UV_WHEEL_MIRROR -eq "$M/pypi/web"); foreach ($src in 'UV_INSTALLER_GITHUB_BASE_URL', 'UNSLOTH_UV_WHEEL_MIRROR') { Run @{ astral = 'blocked' } @{ $src = 'https://corp.example/uv' }; Check "a user $src skips the uv probe" (-not ($script:probed -like '*releases.astral.sh*')) }
    $uvDown = "error: Failed to fetch: ``https://pypi.org/simple/numpy/```n  Caused by: error sending request for url (https://pypi.org/simple/numpy/)"
    $torchDown = "error: Failed to fetch: ``https://download.pytorch.org/whl/cu128/torch/```n  Caused by: operation timed out"
    $npmDown = "npm error network request to https://registry.npmjs.org/react failed, reason: read ECONNRESET"
    $stall = "error: Failed to download distribution`n  Caused by: Failed to download: network timeout"
    $pipDown = "WARNING: Retrying (Retry(total=0)) after connection broken by 'ReadTimeoutError(`"HTTPSConnectionPool(host='pypi.org', port=443): Read timed out.`")': /simple/numpy/"
    $lag = "  cause: Because only unsloth<=2026.9.9 is available and you require unsloth>=2026.9.10, we can conclude that your requirements are unsatisfiable."
    $noVersion = "error: No solution found when resolving dependencies: Because nothing==9 was not found in the package registry (https://pypi.org/simple)"
    foreach ($case in @($uvDown, '', 'pypi'), @($torchDown, 'pypi', 'torch'), @($npmDown, '', 'npm'), @($pipDown, '', 'pypi'), @('npm error code ERESOLVE', 'npm', $null), @($stall, 'pypi', 'pypi'), @($stall, '', $null), @($noVersion, 'pypi', 'unsynced'), @($lag, '', 'unsynced'), @('ERROR: No matching distribution found for unsloth>=2026.9.10', '', 'unsynced'), @("$stall (see https://example.com/)", 'pypi', $null)) {
        Check "failed host: $($case[1]) ran, output '$($case[0].Split("`n")[-1])' -> $($case[2])" ((Get-MirrorFailedHost -Output $case[0] -Ran $case[1]) -eq $case[2])
    }
    function Write-TauriLog {}; function Clear-TauriInstallError {}; function Write-UvDownloadMarker {}; function Write-StudioLine {}
    $script:UnslothVerbose = $false; $script:InstallTorchMirror = $null
    function uv { $script:calls += , "$args"; if ("$env:UV_DEFAULT_INDEX $args" -match 'cernet') { $global:LASTEXITCODE = 0 } else { Write-Error $script:fail; $global:LASTEXITCODE = 1 } }
    function Retry($fail, [scriptblock]$cmd, [switch]$Once) {
        Run @{ torch = 'blocked' }; $env:UNSLOTH_INSTALL_RETRIES = '2'; $env:UNSLOTH_INSTALL_RETRY_DELAY = '0'
        $script:fail = $fail; $script:calls = @(); $script:lines = @()
        if ($Once) { Invoke-InstallCommand -Label t -Command $cmd } else { Invoke-InstallCommandRetry -Label t -Command $cmd }
    }
    $rc = Retry $uvDown { uv pip install numpy }
    Check "pypi transport: retried through the mirror after the default's attempts, which is kept" ($rc -eq 0 -and $script:calls.Count -eq 3 -and $env:UV_DEFAULT_INDEX -eq $pypiMirror -and $script:lines -contains "STEP PyPI failed; retrying through $pypiMirror" -and $env:_UNSLOTH_MIRROR_SPARE -notlike 'pypi|*')
    $script:UnslothVerbose = $true; $rc = Retry $stall { uv pip install numpy } -Once; $script:UnslothVerbose = $false
    Check "a stall naming no host retries on the host that ran, in verbose mode too" ($rc -eq 0 -and $script:calls.Count -eq 2 -and $env:UV_DEFAULT_INDEX -eq $pypiMirror -and $script:lines -contains "STEP PyPI failed; retrying through $pypiMirror")
    foreach ($own in '--torch-backend=auto', '--find-links', 'git+https://github.com/unslothai/unsloth-zoo') {
        $rc = Retry $stall { uv pip install unsloth $own } -Once; Check "a stall in a command that picks its own source ($own) is not retried" ($rc -eq 1 -and $script:calls.Count -eq 1 -and $env:_UNSLOTH_MIRROR_SPARE -like 'pypi|*')
    }
    $rc = Retry $noVersion { uv pip install nothing==9 }; Check "a version not found, with nothing spared behind the mirror, keeps the spare" ($rc -eq 1 -and $script:calls.Count -eq 2 -and $env:_UNSLOTH_MIRROR_SPARE -like 'pypi|*')
    $rc = Retry $uvDown { uv pip install numpy --default-index https://corp.example/simple }; Check "a pinned index is not moved to the PyPI mirror" ($rc -eq 1 -and $script:calls.Count -eq 2)
    function uv { $script:calls += , "$args"; if ($env:UV_INDEX) { $global:LASTEXITCODE = 0 } else { Write-Error $script:fail; $global:LASTEXITCODE = 1 } }
    function Lag([scriptblock]$cmd) { Run @{ pypi = 'slow' }; $script:fail = $lag; $script:calls = @(); $script:lines = @(); Invoke-InstallCommand -Label t -Command $cmd }
    $rc = Lag { uv pip install unsloth --torch-backend=auto }
    Check "a release the mirror lacks reruns once with pypi.org behind it, kept for later steps" ($rc -eq 0 -and $script:calls.Count -eq 2 -and $env:UV_DEFAULT_INDEX -eq 'https://pypi.org/simple' -and $env:UV_INDEX -eq $pypiMirror -and $script:lines -contains 'STEP The PyPI mirror failed; retrying through https://pypi.org/simple' -and $env:_UNSLOTH_MIRROR_SPARE -notmatch 'unsynced')
    foreach ($own in '--no-index', '--index-url') { $rc = Lag { uv pip install unsloth $own }; Check "... not for a command with its own index ($own)" ($rc -eq 1 -and $script:calls.Count -eq 1 -and $env:_UNSLOTH_MIRROR_SPARE -match 'unsynced') }
    function uv { $script:calls += , "$args"; Write-Error $script:fail; $global:LASTEXITCODE = 1 }
    $rc = Retry $uvDown { uv pip install numpy } -Once
    Check "a failed mirror rerun restores the default and spends the spare" ($rc -eq 1 -and $script:calls.Count -eq 2 -and -not $env:UV_DEFAULT_INDEX -and -not $env:PIP_INDEX_URL -and $env:_UNSLOTH_MIRROR_SPARE -notlike 'pypi|*')
    function uv { $script:calls += , "$args"; if ("$args" -match 'download\.pytorch') { Write-Error $script:fail; $global:LASTEXITCODE = 1 } else { $global:LASTEXITCODE = 0 } }
    $TorchIndexUrl = 'https://download.pytorch.org/whl/cu128'; $specs = @('torch', 'https://download.pytorch.org/whl/cu128/xformers.whl')
    $env:_UNSLOTH_MIRROR_SPARE = "pypi|UV_DEFAULT_INDEX=$pypiMirror torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl"; $script:fail = $torchDown; $script:calls = @()
    $rc = Invoke-InstallCommand -Label t -Command { uv pip install @specs --default-index $TorchIndexUrl }
    $rc2 = Invoke-InstallCommand -Label t -Command { uv pip install torch --default-index $TorchIndexUrl }
    Check "torch transport: the command's torch URLs move to the mirror, for later commands too" ($rc -eq 0 -and $rc2 -eq 0 -and $script:calls[1] -eq "pip install torch $M/pytorch/whl/cu128/xformers.whl --default-index $M/pytorch/whl/cu128" -and $script:calls[2] -eq "pip install torch --default-index $M/pytorch/whl/cu128" -and $TorchIndexUrl -eq 'https://download.pytorch.org/whl/cu128' -and $env:_UNSLOTH_MIRROR_SPARE -eq "pypi|UV_DEFAULT_INDEX=$pypiMirror")
    function uv { $script:calls += , "$args"; Write-Error $script:fail; $global:LASTEXITCODE = 1 }
    $script:InstallTorchMirror = $null; $env:_UNSLOTH_MIRROR_SPARE = "torch|UNSLOTH_PYTORCH_MIRROR=$M/pytorch/whl"; Remove-Item Env:UNSLOTH_PYTORCH_MIRROR -ErrorAction SilentlyContinue
    $script:fail = $torchDown; $script:calls = @()
    $rc = Invoke-InstallCommand -Label t -Command { uv pip install torch --default-index https://download.pytorch.org/whl/cu128 }
    Check "torch transport: a failed mirror rerun leaves later torch commands on the default" ($rc -eq 1 -and $script:calls.Count -eq 2 -and -not $env:_UNSLOTH_MIRROR_SPARE -and -not $script:InstallTorchMirror -and -not $env:UNSLOTH_PYTORCH_MIRROR)
    Check "each run starts with no torch mirror from an earlier run" (@($ast.EndBlock.Statements | Where-Object { $_.Extent.Text -eq '$script:InstallTorchMirror = $null' }).Count -eq 1)
    function npm { $script:calls += , "$args"; if ("$args" -match $script:npmOk) { $global:LASTEXITCODE = 0 } else { Write-Error $npmDown; $global:LASTEXITCODE = 1 } }
    foreach ($npmOk in 'npmmirror', 'never') { foreach ($verbose in $false, $true) {
        Run @{ torch = 'blocked' }; $script:calls = @(); $NpmRegistryArgs = @(); $script:npmOk = $npmOk; $script:SetupCommandOutput = ''
        $script:UnslothVerbose = $verbose; $ok = (Invoke-SetupCommand { npm install @NpmRegistryArgs }) -ne 0 -and (Invoke-NpmMirrorRetry); $script:UnslothVerbose = $false
        $spent = $script:calls[-1] -eq 'install --registry https://registry.npmmirror.com' -and $env:_UNSLOTH_MIRROR_SPARE -notmatch '(^| )npm\|'
        if ($npmOk -eq 'npmmirror') { Check "npm transport (verbose $verbose): reinstalled through npmmirror, kept for later npm installs" ($ok -and $spent -and "$NpmRegistryArgs" -eq '--registry https://registry.npmmirror.com' -and $env:UNSLOTH_NPM_REGISTRY -eq 'https://registry.npmmirror.com') }
        else { Check "npm transport (verbose $verbose): a failed npmmirror rerun is not kept" (-not $ok -and $spent -and $NpmRegistryArgs.Count -eq 0 -and -not $env:UNSLOTH_NPM_REGISTRY) }
    } }
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
        Check "${fn}: a source that answered clears the flag the mirror retry needs" (-not $(if ($fn -eq 'Install-UvFromRelease') { $script:UvReleaseUnfetched } else { $script:UvPinnedUnfetched }))
    } }
    function Invoke-WebRequest { throw 'No such host is known.' }
    foreach ($fn in 'Install-UvFromRelease', 'Install-UvFromPinnedRelease') { $null = & $fn; Check "${fn}: no source answering sets the flag the mirror retry needs" ($(if ($fn -eq 'Install-UvFromRelease') { $script:UvReleaseUnfetched } else { $script:UvPinnedUnfetched })) }
} finally {
    foreach ($n in $saved.Keys) { if ($null -eq $saved[$n]) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue } else { Set-Item "Env:$n" $saved[$n] } }
    Remove-Item -LiteralPath $home_ -Recurse -Force -ErrorAction SilentlyContinue
}
if ($failures) { throw "$failures mirror fallback check(s) failed" }
