# Architecture Plan: Fix Issue #7810 - venv rollback atomic move and state recovery

## Architecture Design & Data Flow

### Current Vulnerable Flow:
1. `Start-StudioVenvRollback` initializes `$script:StudioVenvRollbackDir = $candidate`, `$script:StudioVenvRollbackTarget = $ExistingDir`, `$script:StudioVenvRollbackActive = $true`.
2. Calls `Move-Item -LiteralPath $ExistingDir -Destination $candidate -ErrorAction Stop`.
3. If `Move-Item` fails midway on Windows:
   - `$candidate` directory contains part of the files (e.g. `Lib\site-packages\torch`).
   - `$ExistingDir` directory still exists containing remaining files.
   - The `catch` block checks `Test-Path -LiteralPath $ExistingDir` -> `$true`.
   - The `catch` block resets `$script:StudioVenvRollbackActive = $false` and `$script:StudioVenvRollbackDir = $null`.
   - Throw exception -> Rollback tracker is destroyed, environment is split in two and stranded.

### Proposed Safe Flow:
1. In `Start-StudioVenvRollback`:
   - When `Move-Item` throws an exception, check if `$candidate` was created (`Test-Path -LiteralPath $candidate`).
   - If `$candidate` exists and contains files:
     - Execute a recovery attempt to restore files from `$candidate` back to `$ExistingDir`.
     - If recovery successfully restores `$candidate` back to `$ExistingDir` and removes `$candidate`, AND `$ExistingDir` is intact: reset `$script:StudioVenvRollbackActive = $false` and `$script:StudioVenvRollbackDir = $null`.
     - If `$candidate` STILL exists (recovery failed or was partial): DO NOT reset `$script:StudioVenvRollbackActive`. Leave `$script:StudioVenvRollbackActive = $true` so caller's `finally` block or explicit `Restore-StudioVenvRollback` call can execute full restoration.
2. In `Restore-StudioVenvRollback`:
   - If both `$backup` and `$target` exist:
     - Instead of blindly attempting `Remove-Item` on `$target` (which could delete files that weren't moved into `$backup`), move/merge files from `$backup` back into `$target` (e.g., using item-by-item move or directory merge with `Move-Item` / `Copy-Item` / `.NET Directory` fallback), then remove the empty `$backup` directory.

### Impact Assessment
- Modified Files:
  - `install.ps1`: Update `Start-StudioVenvRollback` and `Restore-StudioVenvRollback` functions.
  - `tests/studio/test_install_rollback_lifecycle.ps1`: Add test cases for partial `Move-Item` failure in `Start-StudioVenvRollback` and partial restoration when both `$backup` and `$target` exist.
  - `tests/python/test_windows_python_venv_hardening.py`: Ensure Python installer tests validate venv rollback resilience.

## Verification Plan
1. Run `powershell -File tests/studio/test_install_rollback_lifecycle.ps1` to ensure all existing and new test cases pass.
2. Run `pytest tests/python/test_windows_python_venv_hardening.py` to ensure Python integration tests pass.
