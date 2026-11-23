param(
    [Parameter(Mandatory)][string] $X64,
    [Parameter(Mandatory)][string] $Arm64,
    [Parameter(Mandatory)][string] $Output,
    [Parameter(Mandatory)][string] $Version
)

$ErrorActionPreference = 'Stop'
$nsis = Join-Path $env:LOCALAPPDATA 'tauri/NSIS'
$defines = @(
    "/DX64=$((Resolve-Path -LiteralPath $X64).Path)"
    "/DARM64=$((Resolve-Path -LiteralPath $Arm64).Path)"
    "/DOUTPUT=$([IO.Path]::GetFullPath($Output))"
    "/DVERSION=$(($Version -split '-', 2)[0]).0"
)
& "$nsis/makensis.exe" /V2 @defines "$PSScriptRoot/../../studio/src-tauri/windows/universal.nsi"
if ($LASTEXITCODE -ne 0) { throw "makensis exited $LASTEXITCODE" }
