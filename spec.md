# Specification: Fix Issue #7810 - Failed venv rollback move on Windows splits environment and strands it

## 1. Overview
In `install.ps1`, `Start-StudioVenvRollback` moves an existing venv directory (`$ExistingDir`) to a rollback backup location (`$candidate`) via `Move-Item`. On Windows, `Move-Item` is non-atomic. If `Move-Item` fails mid-transfer (e.g., due to file lock or permission error), part of the venv is moved to `$candidate` while the rest remains in `$ExistingDir`.

Currently, `Start-StudioVenvRollback`'s `catch` block checks `if (Test-Path -LiteralPath $ExistingDir)` and resets `$script:StudioVenvRollbackActive = $false` and `$script:StudioVenvRollbackDir = $null`. This deactivates the rollback tracking even though a partial move occurred and `$candidate` now holds stranded pieces of the venv environment.

## 2. Requirements & Quality Gates

### Requirements
1. **Partial Move Recovery**:
   - If `Move-Item` in `Start-StudioVenvRollback` fails and `$candidate` exists (containing partial venv files), attempt to restore/merge `$candidate` back into `$ExistingDir` or keep `$script:StudioVenvRollbackActive = $true` so `Restore-StudioVenvRollback` can recover it.
   - `$script:StudioVenvRollbackActive` must ONLY be set to `$false` if `$candidate` does not exist (or was fully restored and removed) AND `$ExistingDir` is intact.
2. **Merge/Restore Parity in `Restore-StudioVenvRollback`**:
   - When restoring `$backup` back to `$target`, if both `$backup` and `$target` exist (e.g., due to partial move), items in `$backup` are merged back into `$target` without deleting non-overlapping files in `$target`, ensuring no environment files are stranded or lost.
3. **Unit & Integration Test Coverage**:
   - Create unit tests in `tests/studio/test_install_rollback_lifecycle.ps1` (and `tests/python/test_windows_python_venv_hardening.py` if applicable) that simulate a partial `Move-Item` failure.
   - Achieve >80% code coverage on the rollback logic functions (`Start-StudioVenvRollback`, `Restore-StudioVenvRollback`, `Complete-StudioVenvRollback`, `Remove-StaleStudioVenvRollbacks`).

### Quality Gates Bắt Buộc
- Format: ✅ Code styled cleanly according to PowerShell / Python repository guidelines.
- Lint: ✅ Zero syntax / lint errors (`install.ps1` parses clean with `Parser::ParseFile`).
- Type-check: ✅ Parameter types verified.
- Security-scan: ✅ Zero-AI footprint, clean path validation (`-LiteralPath` everywhere).
- Tests: ✅ All tests pass via `powershell -File tests/studio/test_install_rollback_lifecycle.ps1` and `pytest`.
- Coverage: ✅ >80% coverage on modified rollback code.

## 3. Acceptance Criteria
- [ ] `Start-StudioVenvRollback` handles partial `Move-Item` failure without discarding `$script:StudioVenvRollbackActive` when `$candidate` contains files.
- [ ] `Restore-StudioVenvRollback` successfully restores the environment even if both `$backup` and `$target` exist with partial files.
- [ ] `powershell -File tests/studio/test_install_rollback_lifecycle.ps1` executes with all tests passing (PASS).
- [ ] A clean single commit is prepared on branch `feature/issue-7810-venv-rollback-fix` with DCO sign-off.
