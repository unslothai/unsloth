# Task Breakdown: Issue #7810 Fix

- [ ] Task 1: Write RED Unit Tests in `tests/studio/test_install_rollback_lifecycle.ps1` simulating partial `Move-Item` failure during `Start-StudioVenvRollback`.
- [ ] Task 2: Implement fix in `install.ps1` (`Start-StudioVenvRollback` and `Restore-StudioVenvRollback`) to handle partial moves atomically and preserve rollback state if `$candidate` exists.
- [ ] Task 3: Verify test suite runs GREEN with `powershell -File tests/studio/test_install_rollback_lifecycle.ps1` and `pytest`.
- [ ] Task 4: Run 5-Layer Quality Gate check (FORMAT, LINT, TYPE, SECURE, TEST).
- [ ] Task 5: Create single clean commit with DCO sign-off.
