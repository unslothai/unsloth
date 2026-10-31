; Unsloth NSIS installer hooks

!macro NSIS_HOOK_PREINSTALL
  ; The updater runs this per-user installer unelevated, so an install under Program Files
  ; failed at the first File (#15866): relaunch elevated once; a declined prompt falls through.
  ${If} $UpdateMode = 1
    Push $0
    Push $1
    System::Call 'shell32::IsUserAnAdmin() i .r0'
    ClearErrors
    ${GetOptions} $CMDLINE "/UNSLOTHRELAUNCH" $1
    ${If} ${Errors}
    ${AndIf} $0 = 0
      ClearErrors
      FileOpen $1 "$INSTDIR\.unsloth-update-write-probe" w
      ${If} ${Errors}
        ${GetParameters} $1
        ClearErrors
        ExecShell "runas" "$EXEPATH" "/UNSLOTHRELAUNCH $1 /D=$INSTDIR"
        ${IfNot} ${Errors}
          Quit
        ${EndIf}
      ${Else}
        FileClose $1
        Delete "$INSTDIR\.unsloth-update-write-probe"
      ${EndIf}
    ${EndIf}
    Pop $1
    Pop $0
  ${EndIf}

  ; Windows bundles carry only install.ps1 now. NSIS writes the current resource manifest
  ; and deletes nothing, so an in-place upgrade from a release that shipped both would keep
  ; install.sh forever and make the non-recursive RMDir "$INSTDIR" fail at uninstall.
  ; Gated on our own executable being there: this hook runs before the user can still cancel,
  ; and the directory can be one they picked themselves, so only a directory that already
  ; holds an Unsloth install is ours to tidy.
  ${If} ${FileExists} "$INSTDIR\${MAINBINARYNAME}.exe"
    Delete "$INSTDIR\install.sh"
  ${EndIf}
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  ; Same file, for anyone uninstalling a version that never ran the hook above.
  ${If} ${FileExists} "$INSTDIR\${MAINBINARYNAME}.exe"
    Delete "$INSTDIR\install.sh"
  ${EndIf}
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ; Desktop uninstall must not remove $PROFILE\.unsloth. The CLI/web
  ; installers also use that tree for environments, models, outputs, and
  ; configuration, and there has been no prior public desktop release whose
  ; private state needs cleanup here.
  DetailPrint "Preserved shared Unsloth data at $PROFILE\.unsloth"
!macroend
