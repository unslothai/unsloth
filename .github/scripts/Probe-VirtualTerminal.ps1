# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

<#
.SYNOPSIS
    Measures whether $Host.UI.SupportsVirtualTerminal can replace the native console thunk.

.DESCRIPTION
    install.ps1 and studio/setup.ps1 enable ANSI colour by emitting a P/Invoke stub for
    GetStdHandle / GetConsoleMode / SetConsoleMode and calling it. Those three are half the native
    imports left in install.ps1 and, in studio/setup.ps1, the ONLY reason that file carries the
    reflection-emit helpers, the child-process capability probe and a Device Guard CIM query at all.
    "Defines a dynamic assembly, P/Invokes kernel32, spawns a hidden child interpreter" is the
    densest malicious-looking region in either script, and in setup.ps1 it exists to colour a banner.

    $Host.UI.SupportsVirtualTerminal is the documented replacement: the PowerShell console host
    performs the same enable-VT sequence on its own stdout at startup and reports the outcome.
    Whether it actually agrees on Windows PowerShell 5.1's conhost is a measurement, not a
    deduction, and getting it wrong prints raw escape sequences at a user instead of a banner.

    This script reports BOTH answers side by side so they can be compared. It changes nothing.

.NOTES
    Reading the property inside an ordinary CI step proves nothing. Actions captures stdout, so
    [Console]::IsOutputRedirected is true and both answers are the redirected ones, which is the
    case the installers decide early and never reach the native path for. A real console has to be
    allocated, which is what the -Attached switch and conhost.exe are for; results come back through
    a file because the console's own output goes nowhere a runner can read.
#>

[CmdletBinding()]
param(
    # Required because a console-attached run has no usable stdout.
    [Parameter(Mandatory = $true)][string]$OutFile,

    [switch]$Attached
)

$ErrorActionPreference = 'Continue'

$result = [ordered]@{
    label               = if ($Attached) { 'console (conhost)' } else { 'redirected (Actions step)' }
    hostName            = $Host.Name
    psVersion           = $PSVersionTable.PSVersion.ToString()
    psEdition           = $PSVersionTable.PSEdition
    osCaption           = $null
    osBuild             = $null
    architecture        = $env:PROCESSOR_ARCHITECTURE
    isOutputRedirected  = $null
    supportsVt          = $null
    supportsVtError     = $null
    nativeVt            = $null
    nativeVtError       = $null
    nativeConsoleMode   = $null
}

# Get-ComputerInfo is slow and missing on older 5.1 builds.
try {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
    $result.osCaption = [string]$os.Caption
    $result.osBuild = "$($os.Version).$($os.BuildNumber)"
} catch { $result.osCaption = "THREW: $($_.Exception.GetType().Name)" }

try { $result.isOutputRedirected = [Console]::IsOutputRedirected } catch { $result.isOutputRedirected = "THREW: $($_.Exception.GetType().Name)" }

# Throws under StrictMode on hosts without the property; $false is correct there.
try {
    $result.supportsVt = [bool]$Host.UI.SupportsVirtualTerminal
} catch {
    $result.supportsVtError = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
}

# Emitted rather than compiled: Add-Type on 5.1 runs csc.exe, which this lane must not trigger.
try {
    if (-not ('ProbeVtNative' -as [type])) {
        $asmName = New-Object System.Reflection.AssemblyName 'ProbeVtNativeAsm'
        $asm = [System.Reflection.Emit.AssemblyBuilder]::DefineDynamicAssembly(
            $asmName, [System.Reflection.Emit.AssemblyBuilderAccess]::Run)
        $module = $asm.DefineDynamicModule('ProbeVtNativeMod')
        $typeBuilder = $module.DefineType('ProbeVtNative',
            'Public, Class, AutoClass, AnsiClass, BeforeFieldInit')

        $getStdHandle = $typeBuilder.DefinePInvokeMethod(
            'GetStdHandle', 'kernel32.dll', 'GetStdHandle',
            'Public, Static, PinvokeImpl',
            [System.Reflection.CallingConventions]::Standard,
            [IntPtr], @([int]),
            [System.Runtime.InteropServices.CallingConvention]::Winapi,
            [System.Runtime.InteropServices.CharSet]::Ansi)
        $getStdHandle.SetImplementationFlags(
            $getStdHandle.GetMethodImplementationFlags() -bor [System.Reflection.MethodImplAttributes]::PreserveSig)

        $getConsoleMode = $typeBuilder.DefinePInvokeMethod(
            'GetConsoleMode', 'kernel32.dll', 'GetConsoleMode',
            'Public, Static, PinvokeImpl',
            [System.Reflection.CallingConventions]::Standard,
            [bool], @([IntPtr], [uint32].MakeByRefType()),
            [System.Runtime.InteropServices.CallingConvention]::Winapi,
            [System.Runtime.InteropServices.CharSet]::Ansi)
        $getConsoleMode.DefineParameter(2, 'Out', 'lpMode') | Out-Null
        $getConsoleMode.SetImplementationFlags(
            $getConsoleMode.GetMethodImplementationFlags() -bor [System.Reflection.MethodImplAttributes]::PreserveSig)

        $setConsoleMode = $typeBuilder.DefinePInvokeMethod(
            'SetConsoleMode', 'kernel32.dll', 'SetConsoleMode',
            'Public, Static, PinvokeImpl',
            [System.Reflection.CallingConventions]::Standard,
            [bool], @([IntPtr], [uint32]),
            [System.Runtime.InteropServices.CallingConvention]::Winapi,
            [System.Runtime.InteropServices.CharSet]::Ansi)
        $setConsoleMode.SetImplementationFlags(
            $setConsoleMode.GetMethodImplementationFlags() -bor [System.Reflection.MethodImplAttributes]::PreserveSig)

        $null = $typeBuilder.CreateType()
    }

    $handle = [ProbeVtNative]::GetStdHandle(-11)
    [uint32]$mode = 0
    if (-not [ProbeVtNative]::GetConsoleMode($handle, [ref]$mode)) {
        $result.nativeVt = $false
        $result.nativeVtError = 'GetConsoleMode returned false (not a console handle)'
    } else {
        $result.nativeConsoleMode = ('0x{0:X}' -f $mode)
        $result.nativeVt = [bool][ProbeVtNative]::SetConsoleMode($handle, ($mode -bor 0x0004))
    }
} catch {
    $result.nativeVtError = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
}

$parent = Split-Path -Parent $OutFile
if ($parent -and -not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
}
$result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $OutFile -Encoding UTF8
