# Issue #5093 Fix Draft (for review)

## Goal
Make ROCm source builds of `llama.cpp` robust across distro layouts (including Fedora package-managed ROCm) with the smallest effective change.

## Design Principle
Prefer upstream-aligned defaults and avoid exporting environment variables that can force incorrect path resolution.

## Proposed Minimal Fix
In `studio/setup.sh` ROCm build block:
- stop exporting `ROCM_PATH`
- stop exporting `HIP_PATH`
- keep `HIPCXX` detection as-is
- keep `-DGGML_HIP=ON` and existing `GPU_TARGETS` logic as-is

Why this is minimal and effective:
- Removes the known problematic override (`ROCM_PATH`) implicated by repro.
- Avoids introducing fragile distro-specific path probing for `HIP_DEVICE_LIB_PATH`.
- Keeps all existing backend/arch detection and build orchestration unchanged.
- Aligns behavior closer to upstream `llama.cpp` HIP guidance.

## Patch Sketch
```diff
--- a/studio/setup.sh
+++ b/studio/setup.sh
@@
                 _BUILD_DESC="building (ROCm)"
                 CMAKE_ARGS="$CMAKE_ARGS -DGGML_HIP=ON"
-                export ROCM_PATH="$ROCM_ROOT"
-                export HIP_PATH="$ROCM_ROOT"
 
                 # Use upstream-recommended HIP compiler (not legacy hipcc-as-CXX)
                 if command -v hipconfig &>/dev/null; then
```

## Why Not Auto-Set HIP_DEVICE_LIB_PATH Right Now
Alternative is to keep `ROCM_PATH` and add auto-detection for `HIP_DEVICE_LIB_PATH`.

Reasons to defer that:
- Device-lib path varies by distro packaging and LLVM version directory.
- Requires additional probing logic and error handling.
- More code and more maintenance surface for a narrow problem.

Given the report and current behavior, removing the forced exports is the cleaner first fix.

## Risk Assessment
Low-to-moderate:
- Some environments might currently rely on forced `ROCM_PATH`/`HIP_PATH` exports.
- But affected Fedora setup already breaks with current behavior, and upstream defaults generally work better across layouts.

Mitigation:
- Validate on at least one Ubuntu ROCm environment and one Fedora ROCm environment.
- If regression appears, add a guarded fallback only when CMake HIP test fails.

## Validation by Machine Type
Use whichever AMD Linux access path is available first, then expand coverage.

1. Runpod AMD Linux
- Primary goal: fast first pass to confirm fix unblocks HIP compiler test.
- Run full install flow and confirm `llama-server` builds.

2. AMD Developer Cloud
- Primary goal: independent second environment confirmation.
- If system install is restricted, run from repo checkout path and capture equivalent setup/build logs.

3. Self-hosted Linux AMD (or team lab box)
- Primary goal: final confidence pass before merge.
- Prefer this for repeatability and post-review reruns.

## Suggested Validation Matrix
1. Fedora 43 + distro ROCm/HIP: install succeeds, `llama-server` built.
2. Ubuntu ROCm layout (`/opt/rocm`): install still succeeds.
3. At least two machine types from: Runpod AMD, AMD Dev Cloud, self-hosted AMD Linux.
4. Non-ROCm paths: no behavioral change (CUDA/CPU unaffected).

## Push Gate (When to Push)
Push only after all are true:
1. Repro of pre-fix failure captured on at least one ROCm target.
2. Post-fix success captured on at least one ROCm target.
3. Cross-check success on a second ROCm target or distro.
4. Logs attached: toolchain discovery, gfx detection, and build result.
5. Diff remains minimal (only intended ROCm env handling change).

## Optional Follow-Up (only if needed)
If any environment fails after removing exports, add targeted fallback:
- retry HIP build once with detected `HIP_DEVICE_LIB_PATH`
- keep fallback path behind failure detection, not default path

This preserves a slim default path while covering edge cases.
