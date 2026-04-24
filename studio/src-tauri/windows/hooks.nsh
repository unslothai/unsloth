; Unsloth Studio NSIS installer hooks

!macro NSIS_HOOK_POSTUNINSTALL
  MessageBox MB_YESNO|MB_ICONQUESTION "Remove all Unsloth data ($PROFILE\.unsloth)?$\n$\nThis deletes installed models, training outputs, and configuration." IDNO skip_cleanup
    RMDir /r "$PROFILE\.unsloth"
    DetailPrint "Removed $PROFILE\.unsloth"
  skip_cleanup:
!macroend
