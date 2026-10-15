# Pre-seed tauri-bundler's NSIS toolset and WebView2 bootstrapper so `tauri build` does not
# fetch them unretried. Constants mirror tauri-bundler 2.8.1 (tauri-cli 2.10.1); other CLI
# versions skip. Re-pin from its source when upgrading.

param(
    # dirs::cache_dir()/tauri, the bundler's default tools directory.
    [string] $ToolsDir = (Join-Path $env:LOCALAPPDATA 'tauri'),
    [string] $TauriCliVersion = '',
    [int] $Attempts = 6
)

$ErrorActionPreference = 'Stop'

$expectedCli = 'tauri-cli 2.10.1'
$nsisUrl = 'https://github.com/tauri-apps/binary-releases/releases/download/nsis-3.11/nsis-3.11.zip'
$nsisSha1 = 'EF7FF767E5CBD9EDD22ADD3A32C9B8F4500BB10D'
$nsisZipRoot = 'nsis-3.11'
$utilsUrl = 'https://github.com/tauri-apps/nsis-tauri-utils/releases/download/nsis_tauri_utils-v0.5.3/nsis_tauri_utils.dll'
$utilsSha1 = '75197FEE3C6A814FE035788D1C34EAD39349B860'
$utilsRel = 'Plugins/x86-unicode/additional/nsis_tauri_utils.dll'
$webview2Url = 'https://go.microsoft.com/fwlink/p/?LinkId=2124703'
$requiredFiles = @(
    'makensis.exe',
    'Bin/makensis.exe',
    'Stubs/lzma-x86-unicode',
    'Stubs/lzma_solid-x86-unicode',
    $utilsRel,
    'Include/MUI2.nsh',
    'Include/FileFunc.nsh',
    'Include/x64.nsh',
    'Include/nsDialogs.nsh',
    'Include/WinMessages.nsh',
    'Include/Win/COM.nsh',
    'Include/Win/Propkey.nsh',
    'Include/Win/RestartManager.nsh'
)

if (-not $TauriCliVersion) {
    $TauriCliVersion = (npx --prefix studio tauri --version | Out-String).Trim()
}
if ($TauriCliVersion -ne $expectedCli) {
    Write-Host "::warning::Skipping Tauri tool prefetch: constants are for $expectedCli, found '$TauriCliVersion'. tauri-bundler will download its own tools, unretried."
    exit 0
}

function Save-WithRetry([string] $Uri, [string] $OutFile) {
    for ($i = 1; $i -le $Attempts; $i++) {
        try {
            Invoke-WebRequest -Uri $Uri -OutFile $OutFile -TimeoutSec 120 -UseBasicParsing
            return $true
        } catch {
            Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
            if ($i -eq $Attempts) {
                Write-Host "::warning::Could not download $Uri after $Attempts attempts: $($_.Exception.Message)"
                return $false
            }
            $delay = [math]::Min(60, 5 * [math]::Pow(2, $i - 1))
            Write-Host "Download of $Uri failed (attempt $i/$Attempts): $($_.Exception.Message); retrying in ${delay}s"
            Start-Sleep -Seconds $delay
        }
    }
}

function Test-Sha1([string] $File, [string] $Expected) {
    (Test-Path -LiteralPath $File) -and ((Get-FileHash -LiteralPath $File -Algorithm SHA1).Hash -eq $Expected)
}

function Test-NsisComplete([string] $Dir) {
    foreach ($rel in $requiredFiles) {
        if (-not (Test-Path -LiteralPath (Join-Path $Dir $rel))) { return $false }
    }
    Test-Sha1 (Join-Path $Dir $utilsRel) $utilsSha1
}

New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
$nsisDir = Join-Path $ToolsDir 'NSIS'
$scratch = Join-Path ([IO.Path]::GetTempPath()) "tauri-prefetch-$PID"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null

try {
    if (Test-NsisComplete $nsisDir) {
        Write-Host "NSIS toolset already complete at $nsisDir"
    } else {
        Remove-Item -LiteralPath $nsisDir -Recurse -Force -ErrorAction SilentlyContinue
        $zip = Join-Path $scratch 'nsis.zip'
        $dll = Join-Path $scratch 'nsis_tauri_utils.dll'
        if ((Save-WithRetry $nsisUrl $zip) -and (Save-WithRetry $utilsUrl $dll)) {
            # A hash mismatch is not transient: fail instead of retrying.
            if (-not (Test-Sha1 $zip $nsisSha1)) { throw "nsis-3.11.zip SHA1 mismatch, expected $nsisSha1" }
            if (-not (Test-Sha1 $dll $utilsSha1)) { throw "nsis_tauri_utils.dll SHA1 mismatch, expected $utilsSha1" }
            Expand-Archive -LiteralPath $zip -DestinationPath $scratch -Force
            Move-Item -LiteralPath (Join-Path $scratch $nsisZipRoot) -Destination $nsisDir
            $dllDest = Join-Path $nsisDir $utilsRel
            New-Item -ItemType Directory -Force -Path (Split-Path $dllDest) | Out-Null
            Move-Item -LiteralPath $dll -Destination $dllDest
            if (-not (Test-NsisComplete $nsisDir)) { throw "NSIS toolset at $nsisDir is incomplete after extraction" }
            Write-Host "NSIS toolset ready at $nsisDir"
        } else {
            Remove-Item -LiteralPath $nsisDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    # The bundler reuses this file unchecked, so only a valid Microsoft-signed binary goes here.
    $webview2 = Join-Path $ToolsDir 'MicrosoftEdgeWebview2Setup.exe'
    if (Test-Path -LiteralPath $webview2) {
        Write-Host "WebView2 bootstrapper already present at $webview2"
    } else {
        $tmp = Join-Path $scratch 'MicrosoftEdgeWebview2Setup.exe'
        if (Save-WithRetry $webview2Url $tmp) {
            $sig = Get-AuthenticodeSignature -LiteralPath $tmp
            if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
                throw "WebView2 bootstrapper signature is $($sig.Status) ($($sig.SignerCertificate.Subject))"
            }
            Move-Item -LiteralPath $tmp -Destination $webview2
            Write-Host "WebView2 bootstrapper ready at $webview2"
        }
    }
} finally {
    Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
}
