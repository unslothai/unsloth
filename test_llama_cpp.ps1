<#
.SYNOPSIS
    Test script for llama.cpp compilation and binary validation on Windows.
    Verifies that llama-server was built with CUDA support and can start.

.USAGE
    .\test_llama_cpp.ps1
    .\test_llama_cpp.ps1 -BinaryPath "C:\path\to\llama-server.exe"
#>
param(
    [string]$BinaryPath = ""
)

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  llama.cpp Windows Build Test" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# -- Step 1: Locate the binary ------------------------------------------
Write-Host "1. Locating llama-server binary..." -ForegroundColor Yellow

$SearchPaths = @()

if ($BinaryPath) {
    $SearchPaths += $BinaryPath
}

# Add all known locations
$RepoRoot = $PSScriptRoot
$SearchPaths += Join-Path $RepoRoot "llama.cpp\build\bin\Release\llama-server.exe"
$SearchPaths += Join-Path $RepoRoot "llama.cpp\build\bin\llama-server.exe"
# Legacy: older setup.ps1 built under ~/.unsloth
$SearchPaths += Join-Path $env:USERPROFILE ".unsloth\llama.cpp\build\bin\Release\llama-server.exe"

# Check LLAMA_SERVER_PATH env var
$envPath = $env:LLAMA_SERVER_PATH
if ($envPath) {
    $SearchPaths = @($envPath) + $SearchPaths
}

# Also check system PATH
$systemPath = (Get-Command llama-server -ErrorAction SilentlyContinue)
if ($systemPath) {
    $SearchPaths += $systemPath.Source
}

$FoundBinary = $null
foreach ($p in $SearchPaths) {
    if (Test-Path $p) {
        $FoundBinary = $p
        break
    }
}

if (-not $FoundBinary) {
    Write-Host "   [FAIL] llama-server.exe not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "   Searched locations:" -ForegroundColor Gray
    foreach ($p in $SearchPaths) {
        Write-Host "     - $p" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "   To fix: Run setup.bat to build llama.cpp, or set:" -ForegroundColor Yellow
    Write-Host '   $env:LLAMA_SERVER_PATH = "C:\path\to\llama-server.exe"' -ForegroundColor Yellow
    exit 1
}

Write-Host "   [OK] Found: $FoundBinary" -ForegroundColor Green

# -- Step 2: Check file info --------------------------------------------
Write-Host ""
Write-Host "2. Binary info..." -ForegroundColor Yellow

$fileInfo = Get-Item $FoundBinary
$sizeMB = [math]::Round($fileInfo.Length / 1MB, 1)
Write-Host "   Size:     $sizeMB MB" -ForegroundColor Gray
Write-Host "   Modified: $($fileInfo.LastWriteTime)" -ForegroundColor Gray

# -- Step 3: Check for CUDA symbols ------------------------------------
Write-Host ""
Write-Host "3. Checking for CUDA support..." -ForegroundColor Yellow

# Run with --help or -v and capture output to check for CUDA indicators
$helpOutput = & $FoundBinary --version 2>&1 | Out-String
if (-not $helpOutput) {
    $helpOutput = ""
}

# Check binary dependencies for CUDA DLLs using dumpbin if available
$dumpbin = (Get-Command dumpbin -ErrorAction SilentlyContinue)
$hasCudaDlls = $false

if ($dumpbin) {
    $deps = & dumpbin /dependents $FoundBinary 2>&1 | Out-String
    if ($deps -match "cudart|cublas|cublasLt|nvcuda") {
        $hasCudaDlls = $true
        Write-Host "   [OK] CUDA DLLs found in dependencies (dumpbin)" -ForegroundColor Green
        # Extract CUDA DLL names
        $cudaDlls = ($deps -split "`n") | Where-Object { $_ -match "cuda|cublas|nvcuda" } | ForEach-Object { $_.Trim() }
        foreach ($dll in $cudaDlls) {
            if ($dll) { Write-Host "     - $dll" -ForegroundColor Gray }
        }
    } else {
        Write-Host "   [WARN] No CUDA DLLs found in dependencies!" -ForegroundColor Red
        Write-Host "   This binary was likely compiled WITHOUT -DGGML_CUDA=ON" -ForegroundColor Red
    }
} else {
    # Fallback: check file size (CUDA builds are typically > 50MB)
    if ($sizeMB -gt 40) {
        Write-Host "   [LIKELY OK] Binary is $sizeMB MB (CUDA builds are typically > 50MB)" -ForegroundColor Green
    } else {
        Write-Host "   [WARN] Binary is only $sizeMB MB (CPU-only builds are typically < 30MB)" -ForegroundColor Yellow
        Write-Host "   dumpbin not available for detailed check. Install VS Build Tools." -ForegroundColor Gray
    }
}

# -- Step 4: Quick startup test -----------------------------------------
Write-Host ""
Write-Host "4. Running startup test (will start and immediately stop)..." -ForegroundColor Yellow

# Start llama-server on a random port with no model -- just check it initializes
$testPort = Get-Random -Minimum 49152 -Maximum 65535
$proc = $null

try {
    $proc = Start-Process -FilePath $FoundBinary `
        -ArgumentList "--port", $testPort, "--host", "127.0.0.1" `
        -PassThru -NoNewWindow -RedirectStandardError "$env:TEMP\llama_test_stderr.txt" `
        -RedirectStandardOutput "$env:TEMP\llama_test_stdout.txt"

    # Give it 3 seconds to start
    Start-Sleep -Seconds 3

    # Check if it crashed
    if ($proc.HasExited) {
        $exitCode = $proc.ExitCode
        $stderr = ""
        if (Test-Path "$env:TEMP\llama_test_stderr.txt") {
            $stderr = Get-Content "$env:TEMP\llama_test_stderr.txt" -Raw
        }
        $stdout = ""
        if (Test-Path "$env:TEMP\llama_test_stdout.txt") {
            $stdout = Get-Content "$env:TEMP\llama_test_stdout.txt" -Raw
        }

        $allOutput = "$stdout`n$stderr"

        if ($allOutput -match "failed to initialize CUDA") {
            Write-Host "   [FAIL] CUDA initialization failed!" -ForegroundColor Red
            Write-Host "   The binary was compiled without CUDA support or CUDA drivers are missing." -ForegroundColor Red
            Write-Host ""
            Write-Host "   Rebuild with: cmake -DGGML_CUDA=ON ..." -ForegroundColor Yellow
        } elseif ($allOutput -match "HTTPS is not supported") {
            # This is expected when LLAMA_CURL=OFF -- not a real failure
            Write-Host "   [OK] Binary started (HTTPS warning is expected -- we use local files)" -ForegroundColor Green
        } else {
            Write-Host "   [WARN] Process exited with code $exitCode" -ForegroundColor Yellow
        }

        if ($allOutput.Trim()) {
            Write-Host ""
            Write-Host "   --- Output ---" -ForegroundColor Gray
            $allOutput.Trim().Split("`n") | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
        }
    } else {
        Write-Host "   [OK] llama-server started successfully on port $testPort" -ForegroundColor Green

        # Check CUDA detection from startup output
        Start-Sleep -Seconds 1
        $stderr = ""
        if (Test-Path "$env:TEMP\llama_test_stderr.txt") {
            $stderr = Get-Content "$env:TEMP\llama_test_stderr.txt" -Raw
        }

        if ($stderr -match "CUDA") {
            if ($stderr -match "failed to initialize CUDA") {
                Write-Host "   [FAIL] CUDA init failed at runtime!" -ForegroundColor Red
            } else {
                Write-Host "   [OK] CUDA detected at runtime" -ForegroundColor Green
            }
        }

        # Kill it
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "   Stopped test server." -ForegroundColor Gray
    }
} catch {
    Write-Host "   [ERROR] Could not start llama-server: $_" -ForegroundColor Red
} finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item "$env:TEMP\llama_test_stderr.txt" -ErrorAction SilentlyContinue
    Remove-Item "$env:TEMP\llama_test_stdout.txt" -ErrorAction SilentlyContinue
}

# -- Step 5: Check llama-quantize ---------------------------------------
Write-Host ""
Write-Host "5. Checking llama-quantize..." -ForegroundColor Yellow

$quantizePath = Join-Path (Split-Path $FoundBinary) "llama-quantize.exe"
if (Test-Path $quantizePath) {
    $qSize = [math]::Round((Get-Item $quantizePath).Length / 1MB, 1)
    Write-Host "   [OK] Found: $quantizePath ($qSize MB)" -ForegroundColor Green
} else {
    Write-Host "   [WARN] llama-quantize.exe not found alongside llama-server" -ForegroundColor Yellow
    Write-Host "   GGUF export/quantization won't work without it" -ForegroundColor Yellow
}

# -- Summary ------------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Summary" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   Binary:    $FoundBinary" -ForegroundColor Gray
Write-Host "   Size:      $sizeMB MB" -ForegroundColor Gray
if ($hasCudaDlls) {
    Write-Host "   CUDA:      YES (confirmed via DLL deps)" -ForegroundColor Green
} elseif ($sizeMB -gt 40) {
    Write-Host "   CUDA:      LIKELY (large binary size)" -ForegroundColor Yellow
} else {
    Write-Host "   CUDA:      NO (rebuild with -DGGML_CUDA=ON)" -ForegroundColor Red
}
Write-Host ""

if (-not $hasCudaDlls -and $sizeMB -le 40) {
    Write-Host "To rebuild with CUDA:" -ForegroundColor Yellow
    Write-Host '  1. Delete the build dir: Remove-Item -Recurse -Force "$env:USERPROFILE\.unsloth\llama.cpp\build"' -ForegroundColor Gray
    Write-Host '  2. Re-run: .\setup.bat' -ForegroundColor Gray
    Write-Host ""
}
