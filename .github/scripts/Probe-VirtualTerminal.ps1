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
    # Where to write the JSON result. Required, because a console-attached run has no usable stdout.
    [Parameter(Mandatory = $true)][string]$OutFile,

    # Set when this process already owns a real console, i.e. it was started through conhost.exe.
    # Only affects the reported label; every measurement below is taken the same way either way.
    [switch]$Attached
)

$ErrorActionPreference = 'Continue'

$result = [ordered]@{
    label               = if ($Attached) { 'console (conhost)' } else { 'redirected (Actions step)' }
    hostName            = $Host.Name
    psVersion           = $PSVersionTable.PSVersion.ToString()
    psEdition           = $PSVersionTable.PSEdition
    # The OS build, because the thing being measured is supplied by the OS. A verdict that does not
    # say which ConsoleHost produced it cannot be checked later against the host a user reports.
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

# Get-CimInstance rather than Get-ComputerInfo: the latter is slow and absent on 5.1 before 5.1.14393.
try {
    $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
    $result.osCaption = [string]$os.Caption
    $result.osBuild = "$($os.Version).$($os.BuildNumber)"
} catch { $result.osCaption = "THREW: $($_.Exception.GetType().Name)" }

try { $result.isOutputRedirected = [Console]::IsOutputRedirected } catch { $result.isOutputRedirected = "THREW: $($_.Exception.GetType().Name)" }

# --- The candidate: one property read. ---------------------------------------------------------
# try/catch because a host without the property throws under Set-StrictMode, and the installers run
# under strict mode. $false is the right answer on such a host (the ISE, for one) in any case.
try {
    $result.supportsVt = [bool]$Host.UI.SupportsVirtualTerminal
} catch {
    $result.supportsVtError = "$($_.Exception.GetType().Name): $($_.Exception.Message)"
}

# --- The incumbent: exactly what install.ps1 does today. ---------------------------------------
# Emitted rather than compiled, for the same reason install.ps1 emits it: Add-Type on Windows
# PowerShell 5.1 writes C# to %TEMP% and runs csc.exe, and this script must not be the thing that
# reintroduces a compile into a lane whose whole job is to prove nothing compiles.
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

    # STD_OUTPUT_HANDLE, then ENABLE_VIRTUAL_TERMINAL_PROCESSING, as install.ps1:2131-2135 does.
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
