; Unsloth NSIS installer hooks

!macro NSIS_HOOK_PREINSTALL
  ; Windows bundles used to carry both installers; they now carry only install.ps1.
  ; The install section writes the current resource manifest and deletes nothing, and
  ; the uninstaller deletes only what is in that manifest, so an in-place upgrade from
  ; a pre-split release would leave install.sh behind forever: a Windows machine
  ; keeping a Linux shell script it can never run, and a non-recursive RMDir "$INSTDIR"
  ; that then fails at uninstall. Delete it here, where every upgrade passes.
  Delete "$INSTDIR\install.sh"
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  ; Same file, for anyone uninstalling a version that never ran the hook above.
  Delete "$INSTDIR\install.sh"
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  ; Desktop uninstall must not remove $PROFILE\.unsloth. The CLI/web
  ; installers also use that tree for environments, models, outputs, and
  ; configuration, and there has been no prior public desktop release whose
  ; private state needs cleanup here.
  DetailPrint "Preserved shared Unsloth data at $PROFILE\.unsloth"
!macroend
