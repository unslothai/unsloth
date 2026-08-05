# AUDIT REPORT: PR #7891 - unslothai/unsloth

## 1. Overview & Context
- **Target Repo**: `unslothai/unsloth`
- **PR Number**: `#7891` (Fixes Issue `#7810`)
- **Title**: `Installer: preserve rollback state and merge on partial venv move (#7810)`
- **Branch**: `pr-7891`

## 2. Latest Codex Review Analysis (August 5, 2026 - 3 New Feedback Items)

### 🔴 P1 Issue: Distinguish Partial-Move vs Failed-Reinstall Restore
- **Codex Feedback**:
  > When the old venv was moved to `$backup` successfully and a later install step creates a fresh `$target` before failing, merging `$backup` into that partially-created venv preserves files/packages that exist only in the failed reinstall. This leaves users with a mixed old/new environment rather than restoring the previous environment.
- **Root Cause & Fix**:
  - **Partial Move Case** (`$script:StudioVenvRollbackIsPartialMove = $true`): Old venv files were split between `$target` and `$backup`. We MUST recursively merge `$backup` into `$target` without deleting `$target`'s files.
  - **Failed Reinstall Case** (`$script:StudioVenvRollbackIsPartialMove = $false`): Old venv was fully moved to `$backup`. Any `$target` created afterwards is a failed fresh reinstall. We MUST remove `$target` completely before moving `$backup` back.

### 🟡 P2 Issue 1: Preserve Directory Reparse Points (Symlinks / Junctions)
- **Codex Feedback**:
  > If a rollback backup contains a directory symlink or junction, treating it as an ordinary container recurses through the link instead of moving the link itself.
- **Root Cause & Fix**:
  - In `Restore-StudioVenvDirectoryMerge`, check `if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)`. Treat reparse points as leaf items and move them directly with `Move-Item`.

### 🟡 P2 Issue 2: Abort Rollback on Directory Enumeration Error
- **Codex Feedback**:
  > Using `-ErrorAction SilentlyContinue` during `Get-ChildItem` ignores enumeration errors (ACL/permission issues) and treats the source subtree as empty, prematurely cleaning up `$backup`.
- **Root Cause & Fix**:
  - Use `-ErrorAction Stop` so enumeration failures throw an exception, aborting restore and preserving `$backup`.

## 3. Updated Scope & Action Plan
- **Primary Code File**: [install.ps1](file:///c:/Users/LENOVO/.antigravity-ide/unsloth/install.ps1)
- **Primary Test File**: [test_install_rollback_lifecycle.ps1](file:///c:/Users/LENOVO/.antigravity-ide/unsloth/tests/studio/test_install_rollback_lifecycle.ps1)
