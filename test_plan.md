# TERMINAL TEST PLAN: PR #7891 - unslothai/unsloth

## 1. Scope of Testing
Testing covers the PowerShell installer rollback lifecycle in `install.ps1`, addressing all P1 and P2 items raised by the Codex review bot.

## 2. Test Scenarios & Edge Cases

### Scenario 1: Standard Rollback Creation & Cleanup
- Standard full move, successful replacement, and stale rollback cleanup.

### Scenario 2: Partial Move Tracking Retention & Flag Setup
- Verify `$script:StudioVenvRollbackActive = $true` and `$script:StudioVenvRollbackIsPartialMove = $true` when `Move-Item` fails midway.

### Scenario 3 (P1): Partial Move Restoration with Nested Subdirectories
- Verify recursive merge restores `$backup\Scripts\python.exe` without deleting `$target\Scripts\unsloth.exe`.

### Scenario 4 (P1): Failed Reinstall Restoration (Clean Reinstall Replacement)
- Full move to backup succeeded (`IsPartialMove = $false`). Fresh `$target` created with `new_package.py`.
- Verify `Restore-StudioVenvRollback` wipes `new_package.py` from `$target` and completely restores the original venv from `$backup`.

### Scenario 5 (P2): Directory Reparse Point Preservation
- Create junction/symlink in `$backup`. Verify `Restore-StudioVenvDirectoryMerge` moves the reparse point directly as a link rather than recursing into it.

### Scenario 6 (P2): Enumeration Error Abort
- Verify `Get-ChildItem` uses `-ErrorAction Stop` to fail safely if directory permissions block reading.

## 3. Terminal Execution & RAW LOG Collection
- Run `powershell -File tests/studio/test_install_rollback_lifecycle.ps1`.
