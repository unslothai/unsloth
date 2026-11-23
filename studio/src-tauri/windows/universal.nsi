; Both payloads are already compressed and signed by the native Tauri builds.
Unicode true
RequestExecutionLevel user
SilentInstall silent
AutoCloseWindow true
SetCompress off
Name "Unsloth"
Icon "..\icons\icon.ico"
OutFile "${OUTPUT}"
VIProductVersion "${VERSION}"
VIAddVersionKey "ProductName" "Unsloth"
VIAddVersionKey "FileDescription" "Unsloth Windows installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "Unsloth AI Inc."

!include FileFunc.nsh
!include x64.nsh

Section
  InitPluginsDir
  SetOutPath "$PLUGINSDIR"
  ${If} ${IsNativeARM64}
    File /oname=setup-arm64.exe "${ARM64}"
    StrCpy $0 "$PLUGINSDIR\setup-arm64.exe"
  ${ElseIf} ${IsNativeAMD64}
    File /oname=setup-x64.exe "${X64}"
    StrCpy $0 "$PLUGINSDIR\setup-x64.exe"
  ${Else}
    SetErrorLevel 1633 ; ERROR_INSTALL_PLATFORM_UNSUPPORTED
    Quit
  ${EndIf}

  ${GetParameters} $1
  ; NSIS removes a trailing /D= from $CMDLINE and puts its value in $INSTDIR.
  ${If} $INSTDIR != ""
    StrCpy $1 "$1 /D=$INSTDIR"
  ${EndIf}
  ClearErrors
  ExecWait '"$0" $1' $2
  ${If} ${Errors}
    SetErrorLevel 1603 ; ERROR_INSTALL_FAILURE
  ${Else}
    SetErrorLevel $2
  ${EndIf}
SectionEnd
