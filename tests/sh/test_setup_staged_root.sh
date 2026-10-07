#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A staged update must write nothing outside the stage root, so deletes under $STUDIO_HOME
# are gated on the override. Source-shape asserts: driving the scripts needs a real venv.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"
PASS=0
FAIL=0

check() {
    if [ "$2" = 0 ]; then echo "  PASS: $1"; PASS=$((PASS+1));
    else echo "  FAIL: $1"; FAIL=$((FAIL+1)); fi
}

has() { grep -qF "$2" "$1" && echo 0 || echo 1; }

check "setup.sh reads the stage override" \
    "$(has "$SETUP_SH" 'STAGE_ROOT="${UNSLOTH_STUDIO_STAGE_ROOT:-}"')"
check "setup.sh falls back to STUDIO_HOME" \
    "$(has "$SETUP_SH" 'RUNTIME_ROOT="${STAGE_ROOT:-$STUDIO_HOME}"')"
check "setup.ps1 reads the stage override" \
    "$(has "$SETUP_PS1" '$env:UNSLOTH_STUDIO_STAGE_ROOT')"
check "setup.ps1 falls back to StudioHome" \
    "$(has "$SETUP_PS1" '$RuntimeRoot = if ($StageRoot) { $StageRoot } else { $StudioHome }')"

# Every venv must follow the override, or a staged run installs into the live environment.
for v in VENV_DIR VENV_T5_530_DIR VENV_T5_550_DIR VENV_T5_510_DIR; do
    check "setup.sh points $v at RUNTIME_ROOT" \
        "$(grep -qE "^${v}=\"\\\$RUNTIME_ROOT/" "$SETUP_SH" && echo 0 || echo 1)"
done
for v in VenvDir VenvT5_530Dir VenvT5_550Dir VenvT5_510Dir; do
    check "setup.ps1 points \$$v at RuntimeRoot" \
        "$(grep -qE "^\\\$${v} = Join-Path \\\$RuntimeRoot " "$SETUP_PS1" && echo 0 || echo 1)"
done

# The migration branch is the `elif` (the offline fast path owns the `if`). Anchors are
# asserted non-empty so a reshape fails here instead of grepping an empty string.
_sh_legacy=$(sed -n '/^elif \[ -d "\$STUDIO_HOME\/\.venv_t5" \]; then$/,/^fi$/p' "$SETUP_SH")
check "setup.sh legacy sidecar block found" \
    "$([ -n "$_sh_legacy" ] && echo 0 || echo 1)"
check "setup.sh guards the legacy sidecar removal on STAGE_ROOT" \
    "$(printf '%s' "$_sh_legacy" | grep -qF '[ -z "$STAGE_ROOT" ]' && echo 0 || echo 1)"
check "setup.sh still removes it on a live update" \
    "$(printf '%s' "$_sh_legacy" | grep -qF 'rm -rf "$STUDIO_HOME/.venv_t5"' && echo 0 || echo 1)"

_ps_legacy=$(sed -n '/^} elseif (Test-Path -LiteralPath \$VenvT5Legacy) {$/,/^}$/p' "$SETUP_PS1")
check "setup.ps1 legacy sidecar block found" \
    "$([ -n "$_ps_legacy" ] && echo 0 || echo 1)"
check "setup.ps1 guards the legacy sidecar removal on StageRoot" \
    "$(printf '%s' "$_ps_legacy" | grep -qF 'if (-not $StageRoot)' && echo 0 || echo 1)"
check "setup.ps1 still removes it on a live update" \
    "$(printf '%s' "$_ps_legacy" | grep -qF 'Remove-Item -LiteralPath $VenvT5Legacy -Recurse -Force' && echo 0 || echo 1)"

# The WebView cache belongs to the app that is still running and rendering from it.
check "setup.sh skips the webview cache clear while staging" \
    "$(has "$SETUP_SH" 'if [ -z "$STAGE_ROOT" ] && [ -x "$VENV_DIR/bin/python" ]; then')"
check "setup.ps1 skips the webview cache clear while staging" \
    "$(has "$SETUP_PS1" 'if (-not $StageRoot -and (Test-Path -LiteralPath (Join-Path $VenvDir "Scripts\python.exe") -PathType Leaf)) {')"

check "setup.sh stages the managed Node runtime" \
    "$(has "$SETUP_SH" '_NODE_PARENT="$RUNTIME_ROOT"')"
check "setup.sh stages llama.cpp and whisper.cpp" \
    "$(has "$SETUP_SH" 'UNSLOTH_HOME="$RUNTIME_ROOT"')"
check "setup.sh forwards the staged helper root to whisper.cpp source builds" \
    "$(has "$SETUP_SH" 'env UNSLOTH_HOME="$UNSLOTH_HOME" sh "$_WHISPER_BUILD"')"
check "whisper.cpp source builds honor the managed helper root" \
    "$(has "$SCRIPT_DIR/../../scripts/build_whisper_cpp.sh" '${UNSLOTH_HOME:-}')"
check "setup.sh stages audio.cpp with llama.cpp and whisper.cpp" \
    "$(has "$SETUP_SH" 'AUDIO_CPP_DIR="$UNSLOTH_HOME/audio.cpp"')"
check "setup.sh hands the audio.cpp installer that directory" \
    "$(has "$SETUP_SH" 'install_audio_cpp_prebuilt.py" --install-dir "$AUDIO_CPP_DIR"')"
check "setup.ps1 stages audio.cpp beside the staged llama.cpp" \
    "$(grep -qF '$UnslothHome = Split-Path -Parent $LlamaCppDir' "$SETUP_PS1" && grep -qF '$AudioCppDir = Join-Path $UnslothHome "audio.cpp"' "$SETUP_PS1" && grep -qF '@($AudioCppInstaller, "--install-dir", $AudioCppDir)' "$SETUP_PS1" && echo 0 || echo 1)"
check "setup.sh does not install global uv while staging" \
    "$(has "$SETUP_SH" 'step "uv" "using pip inside the staged environment"')"
check "setup.ps1 stages the managed Node runtime" \
    "$(has "$SETUP_PS1" '$NodeParent = $StageRoot')"
check "setup.ps1 stages llama.cpp and whisper.cpp" \
    "$(grep -qF 'return (Join-Path $StagingRoot "llama.cpp")' "$SETUP_PS1" && grep -qF '$LlamaCppDir = Get-ManagedLlamaCppDir -StagingRoot $StageRoot' "$SETUP_PS1" && grep -qF '$llamaPreflightFailure = Invoke-ManagedLlamaCppPreflight -StagingRoot $StageRoot' "$SETUP_PS1" && echo 0 || echo 1)"
check "setup.ps1 does not persist User PATH while staging" \
    "$(has "$SETUP_PS1" 'Get-Variable -Name StageRoot -ValueOnly -ErrorAction SilentlyContinue')"
check "setup.ps1 leaves vcredist unchanged while staging" \
    "$(has "$SETUP_PS1" 'step "vcredist" "missing; unchanged during staging"')"
check "setup.ps1 leaves long-path policy unchanged while staging" \
    "$(has "$SETUP_PS1" 'step "long paths" "disabled; unchanged during staging"')"
check "setup.ps1 does not install Git while staging" \
    "$(has "$SETUP_PS1" 'Background staging cannot install Git; retry with the foreground updater.')"
check "setup.ps1 preserves foreground Git bootstrap" \
    "$(has "$SETUP_PS1" 'if ($gitNeeded -or -not $StageRoot) {')"
# A staged run must pick the stage root FIRST, ahead of the Studio cache and drive-root fallback.
_tcd="$(awk '/^\$TorchCacheDir = \$null$/{g=1} g{print} g && /^\$env:TORCHINDUCTOR_CACHE_DIR/{exit}' "$SETUP_PS1")"
check "setup.ps1 keeps the staging compiler cache under the stage root" \
    "$(printf '%s' "$_tcd" | grep -qF 'if ($StageRoot) {' \
       && printf '%s' "$_tcd" | grep -qF '$TorchCacheDir = Join-Path $RuntimeRoot "TORCHINDUCTOR_CACHE_DIR"' \
       && echo 0 || echo 1)"
check "setup.ps1 asks about staging before anything else" \
    "$([ "$(printf '%s' "$_tcd" | grep -n 'if ($StageRoot) {' | cut -d: -f1)" -lt \
         "$(printf '%s' "$_tcd" | grep -n 'LongPathsEnabled' | head -1 | cut -d: -f1)" ] \
       && echo 0 || echo 1)"

# A copied venv's activate scripts name the original root, so the staged branch sets PATH itself.
check "setup.sh activates the staged venv without sourcing the copy" \
    "$(has "$SETUP_SH" 'elif [ -n "$STAGE_ROOT" ]; then')"
check "setup.ps1 activates the staged venv without dot-sourcing the copy" \
    "$(has "$SETUP_PS1" 'function Enter-StudioVenv {')"
check "setup.ps1 still asserts the interpreter after activating" \
    "$(has "$SETUP_PS1" 'Assert-VenvActivated -VenvDir $VenvDir')"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
