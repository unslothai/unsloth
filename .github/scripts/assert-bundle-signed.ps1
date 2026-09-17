# Fail if any executable inside a Windows bundle is unsigned. NSIS runs its
# plugin DLLs from $PLUGINSDIR, so a signed installer proves nothing about them.

param(
    # Bundles to unpack; every PE inside is verified.
    [Parameter(Mandatory = $true)][string[]] $Path,
    # 7-Zip, preinstalled on windows-latest.
    [string] $SevenZip = '7z',
    # Known-unsigned leaf names to accept. Keep empty where possible.
    [string[]] $Allow = @()
)

$ErrorActionPreference = 'Continue'
# .ps1/.psm1 included: install.ps1 ships as a bundle resource and runs on first
# launch. Authenticode covers scripts, and Smart App Control checks them.
$exeExtensions = @('.exe', '.dll', '.sys', '.ocx', '.cpl', '.scr', '.ps1', '.psm1')

$unsigned = @()
$checked = 0

foreach ($bundle in $Path) {
    if (-not (Test-Path $bundle -PathType Leaf)) {
        Write-Host "::error::bundle not found: $bundle"
        exit 1
    }
    $name = Split-Path $bundle -Leaf
    Write-Host ''
    Write-Host "=== $name ==="

    $sig = Get-AuthenticodeSignature $bundle
    $checked++
    if ($sig.Status -ne 'Valid') {
        Write-Host "  UNSIGNED  $name  ($($sig.Status))"
        $unsigned += [pscustomobject]@{ Bundle = $name; File = $name; Status = [string]$sig.Status }
    } else {
        Write-Host "  signed    $name  <- $($sig.SignerCertificate.Subject)"
    }

    $dest = Join-Path $env:RUNNER_TEMP ("sigcheck-" + [System.IO.Path]::GetFileNameWithoutExtension($name))
    Remove-Item $dest -Recurse -Force -ErrorAction SilentlyContinue
    & $SevenZip x -y "-o$dest" $bundle | Out-Null
    # 7-Zip leaves a partial tree behind on error, so a created dir proves nothing.
    if ($LASTEXITCODE -ne 0) {
        Write-Host "::error::7-Zip exited $LASTEXITCODE unpacking $name; contents not verified"
        exit 1
    }
    if (-not (Test-Path $dest)) {
        Write-Host "::error::could not unpack $name; cannot verify its contents"
        exit 1
    }

    $inner = Get-ChildItem $dest -Recurse -File |
        Where-Object { $exeExtensions -contains $_.Extension.ToLower() }
    # No hits means 7-Zip dumped PE sections, not that the payload is clean.
    if (-not $inner) {
        Write-Host "::error::no executable payload found inside $name; contents not verified"
        exit 1
    }

    foreach ($f in ($inner | Sort-Object Name)) {
        $checked++
        $s = Get-AuthenticodeSignature $f.FullName
        if ($s.Status -eq 'Valid') {
            Write-Host ("  signed    {0}" -f $f.Name)
        } elseif ($Allow -contains $f.Name) {
            Write-Host ("  ALLOWED   {0}  ({1}) - explicitly accepted as unsigned" -f $f.Name, $s.Status)
        } else {
            # UnknownError = no signature or unbuilt chain; StatusMessage tells which.
            Write-Host ("  UNSIGNED  {0}  ({1})  {2}" -f $f.Name, $s.Status, $s.StatusMessage)
            $unsigned += [pscustomobject]@{ Bundle = $name; File = $f.Name; Status = [string]$s.Status }
        }
    }
    Remove-Item $dest -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ''
Write-Host "checked $checked file(s) across $($Path.Count) bundle(s)"

if (-not $unsigned) {
    Write-Host 'Every executable in every bundle is validly signed.'
    exit 0
}

Write-Host ''
Write-Host '================ UNSIGNED FILES ================'
$unsigned | Format-Table Bundle, File, Status -AutoSize | Out-String | Write-Host
foreach ($u in $unsigned) {
    Write-Host "::error file=$($u.File)::$($u.File) in $($u.Bundle) is $($u.Status) and needs signing"
}
Write-Host ''
Write-Host 'These ship inside the installer and land on the user machine.'
Write-Host 'For NSIS plugin DLLs see the NSISPLUGINS note in windows/installer.nsi.'
exit 1
