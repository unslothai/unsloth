# Issue #5093 Reproduction Report

## Summary
Installing Unsloth Studio via `install.sh` on Fedora 43 with ROCm/HIP installed from distro packages can fail while building `llama.cpp` with HIP enabled.

Observed failure:
- `cmake llama.cpp failed (exit code 1)`
- HIP compiler test fails with: `cannot find ROCm device library; provide its path via '--rocm-path' or '--rocm-device-lib-path'`

## Environment
- OS: Fedora 43 (local)
- GPU stack: ROCm/HIP installed via package manager
- Command path: `curl -fsSL https://unsloth.ai/install.sh | sh`
- Installer mode: local studio install path that builds `llama.cpp` from source

## Test Targets (Machine Types)
Use any of these machine types for reproducibility work. Keep the same pass/fail criteria across all of them.

1. Runpod AMD Linux instance
- Best for fast disposable validation.
- Prefer images where ROCm tools are already available, or install ROCm packages first.
- Good fit for running full install and collecting complete logs.

2. AMD Developer Cloud (notebook-style)
- Good if GPU-backed shell access is available.
- If environment is ephemeral/restricted, run from a repo checkout instead of system-wide installer when needed.
- Ensure shell can run `hipconfig`, `rocminfo`, and CMake/Ninja before executing repro.

3. Self-hosted Linux AMD machine
- Most representative long-term validation target.
- Best option for repeatable regression checks after PR updates.

## Minimum Preconditions (All Targets)
- Linux with supported AMD GPU visible.
- ROCm HIP toolchain available (`hipconfig`, HIP clang, device libs).
- Build tools available (`cmake`, `ninja` or equivalent).
- Enough disk space for source build artifacts.

## Current Installer Behavior (Relevant)
In `studio/setup.sh`, ROCm build setup currently does this before running CMake:
- exports `ROCM_PATH` from detected ROCm root
- exports `HIP_PATH` from detected ROCm root
- exports `HIPCXX` (from `hipconfig -l`, as `.../clang`) when available
- enables `-DGGML_HIP=ON`
- optionally sets `-DGPU_TARGETS=...`

## Repro Steps
1. Verify toolchain on target machine:
   - `hipconfig -R`
   - `hipconfig -l`
   - `rocminfo | grep -oE 'gfx[0-9]{2,4}[a-z]?' | sort -u`
2. Run the install flow that reaches Studio setup and source `llama.cpp` build:
   - `curl -fsSL https://unsloth.ai/install.sh | sh`
   - or run local setup path (`./install.sh --local`) from repo checkout.
3. Observe build phase text similar to `building (ROCm, gfxXXXX)...`.
4. Observe CMake HIP compiler test failure in logs:
   - `cannot find ROCm device library; provide its path via '--rocm-path' or '--rocm-device-lib-path'`

## Logging Requirements (for PR evidence)
Collect and attach for each target used:
- Full installer log from start to failure/success.
- Output of `hipconfig -R` and `hipconfig -l`.
- Output snippet showing detected `gfx` target(s).
- Exact CMake HIP compiler error block (if reproducing failure).

## Expected vs Actual
Expected:
- HIP toolchain config succeeds and `llama-server` builds.

Actual:
- CMake HIP compiler test fails, and source build aborts.

## Validation Clue from Reporter
The reporter confirmed upstream `llama.cpp` HIP build succeeds on same machine when using upstream-style command:
- `HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake ... -DGGML_HIP=ON ...`

Reporter also found:
- Removing `export ROCM_PATH=...` in Unsloth setup made install succeed.
- Keeping `ROCM_PATH` can be made to work if `HIP_DEVICE_LIB_PATH` is also set explicitly.

## Root-Cause Hypothesis
On distro ROCm layouts (Fedora example), exporting `ROCM_PATH` unconditionally can steer HIP/CMake/clang device-lib discovery into a path that does not include the actual amdgcn bitcode location. Since installer does not also provide `HIP_DEVICE_LIB_PATH`, compiler test can fail.

In short:
- setting `ROCM_PATH` alone can be harmful on some layouts
- setting neither (or setting both correctly) works

## Scope
Issue is specific to source `llama.cpp` HIP builds in Studio setup path. It does not imply ROCm is broken on the system, since upstream `llama.cpp` build works.
