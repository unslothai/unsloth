; Unsloth Studio NSIS installer hooks

!macro NSIS_HOOK_POSTUNINSTALL
  ; Desktop uninstall must not remove $PROFILE\.unsloth. The CLI/web
  ; installers also use that tree for environments, models, outputs, and
  ; configuration, and there has been no prior public desktop release whose
  ; private state needs cleanup here.
  DetailPrint "Preserved shared Unsloth data at $PROFILE\.unsloth"
!macroend
