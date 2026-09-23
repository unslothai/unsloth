#!/usr/bin/env pwsh
# install.ps1's mirror fallback (synced into setup.ps1): the probe against a local server, and the
# env vars each probe outcome exports. Run: pwsh -NoProfile -File tests/studio/test_mirror_fallback.ps1

$ErrorActionPreference = "Stop"
$installPath = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1"))).Path
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { throw "install.ps1 has parse errors" }
$setupAst = [System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1"))).Path, [ref]$tokens, [ref]$errors)
foreach ($pair in @($ast, 'Test-MirrorConfigured'), @($ast, 'Invoke-MirrorProbe'), @($ast, 'Invoke-MirrorFallback'), @($ast, 'Install-UvFromRelease'), @($setupAst, 'Install-UvFromPinnedRelease'), @($setupAst, 'Get-UvInstallDir')) {
    $fn = $pair[0].FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $pair[1] }, $true)
    Invoke-Expression $fn[0].Extent.Text
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" } else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# ok answers at once, slow sends its status and then stalls the body, 404 (or no User-Agent, as on CERNET) is an HTTP error.
$server = @'
import http.server, sys, time
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(404 if self.path == "/404" or not self.headers.get("User-Agent") else 200); self.send_header("Content-Length", "10"); self.end_headers()
        self.wfile.write(b"x"); self.wfile.flush()
        time.sleep(5 if self.path == "/slow" else 0); self.wfile.write(b"x" * 9)
    def log_message(self, *a): pass
s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); print(s.server_port, flush=True); s.serve_forever()
'@
# Windows' python3 is often the Microsoft Store stub, which prints an install hint instead of running.
$python = if ($env:OS -ne 'Windows_NT' -and (Get-Command python3 -ErrorAction SilentlyContinue)) { 'python3' } else { 'python' }
$psi = New-Object System.Diagnostics.ProcessStartInfo $python, "-c `"exec(__import__('sys').stdin.read())`""
$psi.RedirectStandardInput = $true; $psi.RedirectStandardOutput = $true; $psi.UseShellExecute = $false
$proc = [System.Diagnostics.Process]::Start($psi)
$proc.StandardInput.Write($server); $proc.StandardInput.Close()
$base = "http://127.0.0.1:$($proc.StandardOutput.ReadLine())"
try {
    $protocol = [System.Net.ServicePointManager]::SecurityProtocol
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.SecurityProtocolType]::Tls
    $r = Invoke-MirrorProbe -Urls @("$base/ok", "$base/slow", "$base/404", "http://127.0.0.1:9/") -Seconds 2
    Check "probe: a pinned legacy TLS setting is restored" ([System.Net.ServicePointManager]::SecurityProtocol -eq 'Tls')
    [System.Net.ServicePointManager]::SecurityProtocol = $protocol
    Check "probe: fast 200 is ok" ($r["$base/ok"] -eq 'ok')
    Check "probe: status but stalled body is slow" ($r["$base/slow"] -eq 'slow')
    Check "probe: HTTP error is blocked" ($r["$base/404"] -eq 'blocked')
    Check "probe: refused connection is blocked" ($r["http://127.0.0.1:9/"] -eq 'blocked')
} finally { $proc.Kill() }

$names = '_UNSLOTH_MIRROR_PROBED', 'UNSLOTH_MIRROR_FALLBACK', 'UV_INDEX', 'UV_DEFAULT_INDEX', 'UV_INDEX_URL', 'UV_INDEX_STRATEGY', 'PIP_INDEX_URL',
    'PIP_EXTRA_INDEX_URL', 'UNSLOTH_PYTORCH_MIRROR', 'UNSLOTH_NODE_MIRROR', 'UNSLOTH_NPM_REGISTRY', 'UV_PYTHON_INSTALL_MIRROR', 'UV_CONFIG_FILE', 'PIP_CONFIG_FILE',
    'UNSLOTH_UV_WHEEL_MIRROR', 'UV_INSTALLER_GITHUB_BASE_URL', 'UV_INSTALL_DIR', 'UV_NO_MODIFY_PATH'
$saved = @{}; foreach ($n in $names + 'APPDATA', 'USERPROFILE', 'ProgramData', 'PATH') { $saved[$n] = [Environment]::GetEnvironmentVariable($n) }
$home_ = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-mirror-" + [guid]::NewGuid())
$pipIni = Join-Path $home_ 'AppData/pip/pip.ini'
New-Item -ItemType Directory -Force -Path (Split-Path $pipIni) | Out-Null
$env:APPDATA = Join-Path $home_ 'AppData'; $env:USERPROFILE = $home_; $env:ProgramData = Join-Path $home_ 'ProgramData'
function step($l, $v) { $script:lines += "STEP $v" }
function substep($m) { $script:lines += "SUBSTEP $m" }
# Default hosts answer from $script:outcome by host name; mirrors answer ok unless listed in $script:deadMirror.
function Invoke-MirrorProbe([string[]]$Urls) {
    $script:probed += $Urls; $out = @{}
    foreach ($u in $Urls) { $h = ([uri]$u).Host; $out[$u] = if ($script:outcome.ContainsKey($h)) { $script:outcome[$h] } elseif ($script:deadMirror -contains $h) { 'blocked' } else { 'ok' } }
    return $out
}
function Run($outcome, $envs = @{}, $deadMirror = @()) {
    foreach ($n in $names) { Remove-Item "Env:$n" -ErrorAction SilentlyContinue }
    foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
    $script:outcome = $outcome; $script:deadMirror = $deadMirror; $script:lines = @(); $script:probed = @()
    Invoke-MirrorFallback
}
$M = 'https://mirrors.cernet.edu.cn'
try {
    Run @{}; Check "healthy hosts export and print nothing" (-not $env:UV_DEFAULT_INDEX -and -not $env:UNSLOTH_NPM_REGISTRY -and $script:lines.Count -eq 0)
    Run @{ 'pypi.org' = 'blocked' }
    Check "blocked pypi: mirror is the only uv/pip index" ($env:UV_DEFAULT_INDEX -eq "$M/pypi/web/simple" -and $env:PIP_INDEX_URL -eq "$M/pypi/web/simple" -and -not $env:UV_INDEX -and -not $env:UV_INDEX_STRATEGY -and -not $env:PIP_EXTRA_INDEX_URL)
    Check "blocked pypi: says so and names the opt-out" ($script:lines -contains "STEP pypi.org is blocked; using $M/pypi/web/simple" -and $script:lines[-1] -like 'SUBSTEP Set UNSLOTH_MIRROR_FALLBACK=0*')
    Run @{ 'pypi.org' = 'slow' }
    Check "slow pypi: mirror first, pypi.org second, unsafe-first-match" ($env:UV_INDEX -eq "$M/pypi/web/simple" -and $env:UV_DEFAULT_INDEX -eq 'https://pypi.org/simple' -and $env:UV_INDEX_STRATEGY -eq 'unsafe-first-match' -and $env:PIP_INDEX_URL -eq "$M/pypi/web/simple" -and $env:PIP_EXTRA_INDEX_URL -eq 'https://pypi.org/simple')
    Run @{ 'pypi.org' = 'slow' } @{ UV_INDEX_STRATEGY = 'first-index' }; Check "a user index strategy is kept" ($env:UV_INDEX_STRATEGY -eq 'first-index')
    Run @{ 'pypi.org' = 'blocked' } @{} @('mirrors.cernet.edu.cn'); Check "a dead mirror changes and prints nothing" (-not $env:UV_DEFAULT_INDEX -and $script:lines.Count -eq 0)
    Run @{ 'download.pytorch.org' = 'blocked'; 'nodejs.org' = 'slow'; 'registry.npmjs.org' = 'blocked' }
    Check "torch, node and npm mirrors; healthy pypi untouched" ($env:UNSLOTH_PYTORCH_MIRROR -eq "$M/pytorch/whl" -and $env:UNSLOTH_NODE_MIRROR -eq "$M/nodejs-release" -and $env:UNSLOTH_NPM_REGISTRY -eq 'https://registry.npmmirror.com' -and -not $env:PIP_INDEX_URL)
    Check "healthy pypi skips its mirror probe" (-not ($script:probed -like "$M/pypi/*"))
    Run @{ 'pypi.org' = 'blocked' } @{ UV_INDEX_URL = 'https://corp.example/simple' }; Check "user uv index env: only pip falls back" (-not $env:UV_DEFAULT_INDEX -and $env:PIP_INDEX_URL)
    New-Item -ItemType Directory -Force -Path (Join-Path $env:APPDATA 'uv') | Out-Null
    foreach ($toml in "[[index]]`nurl = 'https://corp.example/simple'", "index = [{ url = 'https://corp.example/simple', default = true }]", "pip.index-url = 'https://corp.example/simple'") {
        Set-Content -Path (Join-Path $env:APPDATA 'uv/uv.toml') -Value $toml
        Run @{ 'pypi.org' = 'blocked' }; Check "uv.toml $($toml.Split("`n")[0]): only pip falls back" (-not $env:UV_DEFAULT_INDEX -and $env:PIP_INDEX_URL)
    }
    $pbs = 'https://registry.npmmirror.com/-/binary/python-build-standalone'
    Run @{ 'releases.astral.sh' = 'slow' }; Check "slow Python builds: uv downloads them from npmmirror" ($env:UV_PYTHON_INSTALL_MIRROR -eq $pbs -and $script:lines -contains "STEP releases.astral.sh (Python builds) is slow; using $pbs" -and -not $env:UV_DEFAULT_INDEX)
    Run @{ 'releases.astral.sh' = 'blocked' } @{ UV_PYTHON_INSTALL_MIRROR = 'https://corp.example/pbs' }; Check "a user Python mirror is kept and not probed" ($env:UV_PYTHON_INSTALL_MIRROR -eq 'https://corp.example/pbs' -and -not ($script:probed -like '*python-build*'))
    Set-Content -Path (Join-Path $env:APPDATA 'uv/uv.toml') -Value "python-install-mirror = 'https://corp.example/pbs'"
    Run @{ 'releases.astral.sh' = 'blocked'; 'pypi.org' = 'blocked' }; Check "uv.toml python-install-mirror: kept, index still falls back" (-not $env:UV_PYTHON_INSTALL_MIRROR -and $env:UV_DEFAULT_INDEX)
    Remove-Item -LiteralPath (Join-Path $env:APPDATA 'uv/uv.toml')
    Set-Content -Path $pipIni -Value "[global]`nindex-url = https://corp.example/simple"
    Run @{ 'pypi.org' = 'blocked' }; Check "pip.ini index: only uv falls back" (-not $env:PIP_INDEX_URL -and $env:UV_DEFAULT_INDEX)
    Set-Content -Path $pipIni -Value "[global]`ntimeout = 60"
    Run @{ 'pypi.org' = 'blocked' }; Check "pip.ini without an index still falls back" ($env:PIP_INDEX_URL -eq "$M/pypi/web/simple")
    Run @{ 'download.pytorch.org' = 'blocked' } @{ UNSLOTH_TORCH_INDEX_URL = 'https://corp.example/whl/cu128' }; Check "a pinned torch index is not overridden" (-not $env:UNSLOTH_PYTORCH_MIRROR)
    Run @{ 'registry.npmjs.org' = 'blocked' } @{ UNSLOTH_NPM_REGISTRY = 'https://corp.example/npm/' }; Check "a user npm registry is kept and not probed" ($env:UNSLOTH_NPM_REGISTRY -eq 'https://corp.example/npm/' -and -not ($script:probed -like '*npm*'))
    Run @{ 'pypi.org' = 'blocked' } @{ UNSLOTH_MIRROR_FALLBACK = 'Off' }; Check "UNSLOTH_MIRROR_FALLBACK=Off probes nothing" ($script:probed.Count -eq 0)
    Run @{ 'pypi.org' = 'blocked' } @{ _UNSLOTH_MIRROR_PROBED = '1' }; Check "a parent installer's probe is not repeated" ($script:probed.Count -eq 0)
    Run @{ 'releases.astral.sh' = 'blocked' }; Check "blocked uv releases: the pinned wheel comes from the PyPI mirror" ($env:UNSLOTH_UV_WHEEL_MIRROR -eq "$M/pypi/web"); foreach ($src in 'UV_INSTALLER_GITHUB_BASE_URL', 'UNSLOTH_UV_WHEEL_MIRROR') { Run @{ 'releases.astral.sh' = 'blocked' } @{ $src = 'https://corp.example/uv' }; Check "a user $src skips the uv probe" (-not ($script:probed -like '*/uv/*')) }
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
