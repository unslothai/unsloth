<#
.SYNOPSIS
    Extracts a single `function NAME { ... }` block from a PowerShell script by
    brace-matching, WITHOUT executing the script.

.DESCRIPTION
    studio/setup.ps1 is a top-level executing installer (it runs install steps at
    load), so it cannot be dot-sourced directly in a test. This helper pulls just
    the requested function's source text out of the file so a test can dot-source
    ONLY that function.

    Brace matching is naive (it counts '{' / '}' without a full tokenizer). It is
    safe for the pure helper functions targeted here because their bodies contain
    only balanced braces (e.g. `${env:ProgramFiles(x86)}` is self-balanced) and no
    here-strings/comments with stray unbalanced braces.

.EXAMPLE
    $src = Get-FunctionSource -Path studio/setup.ps1 -Name Get-VcBuildCustomizationsDir
    . ([scriptblock]::Create($src))   # defines the function in the current scope
#>
function Get-FunctionSource {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Name
    )

    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $text = Get-Content -Raw -LiteralPath $Path
    if ([string]::IsNullOrEmpty($text)) { return $null }

    # Match "function <Name>" at the start of a line (multiline, case-insensitive).
    $pattern = "(?im)^\s*function\s+$([regex]::Escape($Name))\b"
    $m = [regex]::Match($text, $pattern)
    if (-not $m.Success) { return $null }

    # Locate the opening brace at/after the match.
    $braceStart = $text.IndexOf('{', $m.Index)
    if ($braceStart -lt 0) { return $null }

    # Walk braces to the matching close.
    $depth = 0
    $end = -1
    for ($i = $braceStart; $i -lt $text.Length; $i++) {
        $c = $text[$i]
        if ($c -eq '{') { $depth++ }
        elseif ($c -eq '}') {
            $depth--
            if ($depth -eq 0) { $end = $i; break }
        }
    }
    if ($end -lt 0) { return $null }

    return $text.Substring($m.Index, $end - $m.Index + 1)
}
