#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# install.sh persists a PATH entry by appending a line to a shell rc file (or a
# fish_add_path line to ~/.config/fish/conf.d/unsloth.fish). Written as a PREPEND,
# that line lands after whatever conda's own init block put in the same file, so
# every later shell resolves out of our directory before the active conda
# environment's -- an ordering that outlives the activation it was created under.
# That is the POSIX half of #5871, whose Windows half is the registry write guarded
# in tests/python/test_installer_conda_path_guard.py.
#
# The properties, in the order they are checked:
#
#   1. Outside conda the written line is unchanged: PATH prepend, exactly as before.
#   2. Inside an active conda environment the same directory is registered at the
#      BACK, so conda keeps priority.
#   3. Either spelling counts as already present. Without that, a machine that
#      installed once inside conda and once outside it collects two lines for the
#      same directory, and a fish user collects two fish_add_path calls.
#   4. The advice printed when the rc file cannot be written matches the line that
#      would have been written, or the manual fix reintroduces the defect by hand.
#
# studio/setup.sh carries a third copy of the same write and is covered at the end of
# this file, because it runs in its own process right after install.sh: a guard in
# install.sh alone leaves the breakage reachable from the next step of the same install.
#
# The functions are extracted from the installers and run, rather than restated here, so
# these cases track them instead of a copy of them.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_SH="$REPO_ROOT/install.sh"

echo "=== test_install_conda_path_guard ==="

# Both persisters plus the regex they share, delimited by the file's own top-level
# function braces. Asserted non-empty below, so a rename is loud rather than vacuous.
extract_function() {
    awk -v name="$1" '
        $0 == name "() {" { grab = 1 }
        grab { print }
        grab && $0 == "}" { exit }
    ' "$INSTALL_SH"
}

PATH_LINE_RE_SRC=$(grep -n "^_PATH_LINE_RE=" "$INSTALL_SH" | head -n1 | cut -d: -f2-)
LOGIN_FN=$(extract_function _persist_login_path_dir)
FISH_FN=$(extract_function _persist_fish_path_dir)
# The predicate both writers ask. Extracted rather than restated, so a change to WHICH
# environment variables count as an active conda reaches these cases.
CONDA_FN=$(extract_function _unsloth_conda_env_active)

for _chunk_label in _PATH_LINE_RE _persist_login_path_dir _persist_fish_path_dir _unsloth_conda_env_active; do
    case "$_chunk_label" in
        _PATH_LINE_RE)             _chunk="$PATH_LINE_RE_SRC" ;;
        _persist_login_path_dir)   _chunk="$LOGIN_FN" ;;
        _unsloth_conda_env_active) _chunk="$CONDA_FN" ;;
        *)                         _chunk="$FISH_FN" ;;
    esac
    if [ -z "$_chunk" ]; then
        echo "  FAIL: could not extract $_chunk_label from install.sh"
        exit 1
    fi
done

# The two reporters the functions call. `step` and `substep` print in install.sh; here
# they are captured so assertion 4 can read the advice that was offered.
HARNESS=$(cat <<'EOF'
step()    { echo "STEP $1 $2"; }
substep() { echo "SUBSTEP $1"; }
C_WARN=""
EOF
)

# One run of _persist_login_path_dir against a throwaway HOME.
# $1 rc file contents ("" for an empty file), $2 CONDA_PREFIX ("" for none).
run_login() {
    WORK=$(mktemp -d)
    RC_FILE=${RC_NAME:+$WORK/$RC_NAME}
    printf '%s' "$1" > "${RC_FILE:-$WORK/.bashrc}"
    (
        eval "$HARNESS"
        eval "$PATH_LINE_RE_SRC"
        eval "$CONDA_FN"
        eval "$LOGIN_FN"
        HOME="$WORK"
        SHELL="${4:-/bin/bash}"
        unset CONDA_PREFIX CONDA_DEFAULT_ENV
        # $3 names WHICH variable carries the activation: a shell hook that exports only
        # CONDA_DEFAULT_ENV still leaves the caller inside conda's PATH ordering, which is
        # why install.ps1 reads both and these installers have to agree with it.
        if [ -n "$2" ]; then eval "${3:-CONDA_PREFIX}=\"\$2\"; export ${3:-CONDA_PREFIX}"; fi
        _persist_login_path_dir "$WORK/.local/bin" '$HOME/.local/bin' "~/.local/bin" '\.local/bin' "${RC_FILE:-$WORK/.bashrc}"
    )
    RC_CONTENTS=$(cat "${RC_FILE:-$WORK/.bashrc}")
    rm -rf "$WORK"
}

# 1. Outside conda: unchanged behaviour, a prepend.
run_login "" ""
assert_contains "no conda: writes a PATH prepend" "$RC_CONTENTS" 'export PATH="$HOME/.local/bin:$PATH"'

# 2. Inside an active conda environment: the same directory, at the back.
run_login "" "/opt/anaconda3"
assert_contains "active conda: writes a PATH append" "$RC_CONTENTS" 'export PATH="$PATH:$HOME/.local/bin"'
assert_not_contains "active conda: does not prepend" "$RC_CONTENTS" 'export PATH="$HOME/.local/bin:$PATH"'

# 3. Either spelling is already-present. Both directions, because a machine can meet
#    the installer inside conda first and outside it later, or the reverse.
run_login 'export PATH="$PATH:$HOME/.local/bin"' ""
assert_not_contains "append already there: no prepend added" "$RC_CONTENTS" 'export PATH="$HOME/.local/bin:$PATH"'
assert_eq "append already there: exactly one PATH line" "1" "$(printf '%s\n' "$RC_CONTENTS" | grep -c 'local/bin')"

run_login 'export PATH="$HOME/.local/bin:$PATH"' "/opt/anaconda3"
assert_eq "prepend already there: exactly one PATH line" "1" "$(printf '%s\n' "$RC_CONTENTS" | grep -c 'local/bin')"

# 4. An unwritable rc file: the advice has to be the conda-safe line, not the prepend.
#    Chmod, not a directory: the failure path under test is the append redirect failing.
WORK=$(mktemp -d)
: > "$WORK/.bashrc"
chmod 400 "$WORK/.bashrc"
ADVICE=$(
    eval "$HARNESS"
    eval "$PATH_LINE_RE_SRC"
    eval "$CONDA_FN"
    eval "$LOGIN_FN"
    HOME="$WORK"
    SHELL=/bin/bash
    CONDA_PREFIX="/opt/anaconda3"; export CONDA_PREFIX
    _persist_login_path_dir "$WORK/.local/bin" '$HOME/.local/bin' "~/.local/bin" '\.local/bin' "$WORK/.bashrc" 2>&1
)
chmod 600 "$WORK/.bashrc"
if printf '%s' "$ADVICE" | grep -q 'could not write'; then
    assert_contains "unwritable rc: advice is the append form" "$ADVICE" 'export PATH="$PATH:$HOME/.local/bin"'
else
    # Running as root, where a 400 file is still writable. The case is about the advice
    # text, so skipping it loudly beats asserting nothing.
    ok "unwritable rc: skipped (this user can write a mode-400 file)"
fi
rm -rf "$WORK"

# 5. CONDA_DEFAULT_ENV alone. A shell hook can export it without CONDA_PREFIX, and the
#    caller is still inside conda's PATH ordering. install.ps1 has always read both; the
#    POSIX writers used to read only CONDA_PREFIX, so the same user got a prepend on Linux
#    and an append on Windows.
run_login "" "myenv" CONDA_DEFAULT_ENV
assert_contains "CONDA_DEFAULT_ENV alone: writes a PATH append" "$RC_CONTENTS" 'export PATH="$PATH:$HOME/.local/bin"'
assert_not_contains "CONDA_DEFAULT_ENV alone: does not prepend" "$RC_CONTENTS" 'export PATH="$HOME/.local/bin:$PATH"'

# 6. zsh, which is the default shell on macOS and lands in a different rc file. Same
#    writer, so the same guard has to apply to it.
RC_NAME=.zshrc run_login "" "/opt/anaconda3" CONDA_PREFIX /bin/zsh
assert_contains "zsh rc, active conda: appends" "$RC_CONTENTS" 'export PATH="$PATH:$HOME/.local/bin"'
RC_NAME=.zshrc run_login "" "" CONDA_PREFIX /bin/zsh
assert_contains "zsh rc, no conda: prepends" "$RC_CONTENTS" 'export PATH="$HOME/.local/bin:$PATH"'

# ── fish ───────────────────────────────────────────────────────────────────────────
# fish reads none of the POSIX rc files, so it has its own writer and its own defect:
# fish_add_path prepends by default and -a is the documented append.
run_fish() {
    WORK=$(mktemp -d)
    mkdir -p "$WORK/.config/fish/conf.d"
    printf '%s' "$1" > "$WORK/.config/fish/conf.d/unsloth.fish"
    (
        eval "$HARNESS"
        eval "$CONDA_FN"
        eval "$FISH_FN"
        HOME="$WORK"
        unset CONDA_PREFIX CONDA_DEFAULT_ENV
        if [ -n "$2" ]; then eval "${3:-CONDA_PREFIX}=\"\$2\"; export ${3:-CONDA_PREFIX}"; fi
        _persist_fish_path_dir "$WORK/.local/bin" "~/.local/bin"
    )
    FISH_CONTENTS=$(cat "$WORK/.config/fish/conf.d/unsloth.fish")
    FISH_DIR="$WORK/.local/bin"
    rm -rf "$WORK"
}

run_fish "" ""
assert_contains "fish, no conda: prepends" "$FISH_CONTENTS" "fish_add_path '$FISH_DIR'"

run_fish "" "/opt/anaconda3"
assert_contains "fish, active conda: appends" "$FISH_CONTENTS" "fish_add_path -a '$FISH_DIR'"

run_fish "" "myenv" CONDA_DEFAULT_ENV
assert_contains "fish, CONDA_DEFAULT_ENV alone: appends" "$FISH_CONTENTS" "fish_add_path -a '$FISH_DIR'"

# Idempotence across both spellings, the same way as the POSIX arm.
WORK=$(mktemp -d)
mkdir -p "$WORK/.config/fish/conf.d"
printf "fish_add_path -a '%s/.local/bin'\n" "$WORK" > "$WORK/.config/fish/conf.d/unsloth.fish"
(
    eval "$HARNESS"
    eval "$CONDA_FN"
    eval "$FISH_FN"
    HOME="$WORK"
    unset CONDA_PREFIX CONDA_DEFAULT_ENV
    _persist_fish_path_dir "$WORK/.local/bin" "~/.local/bin"
) >/dev/null
assert_eq "fish: append already there, no second line" "1" \
    "$(grep -c 'fish_add_path' "$WORK/.config/fish/conf.d/unsloth.fish")"
rm -rf "$WORK"

# ── studio/setup.sh ────────────────────────────────────────────────────────────────
# The third copy of the same write. setup.sh runs after install.sh, in its own process,
# and persists the uv directory into the same rc files, so a guard in install.sh alone
# leaves the reported breakage reachable by the very next step of the same install.
SETUP_SH="$REPO_ROOT/studio/setup.sh"
extract_setup_function() {
    awk -v name="$1" '
        $0 == name "() {" { grab = 1 }
        grab { print }
        grab && $0 == "}" { exit }
    ' "$SETUP_SH"
}
SETUP_FN=$(extract_setup_function _setup_persist_uv_path)
SETUP_HAS_DIR_FN=$(extract_setup_function _setup_path_has_dir)
SETUP_CONDA_FN=$(extract_setup_function _unsloth_conda_env_active)
if [ -z "$SETUP_FN" ] || [ -z "$SETUP_HAS_DIR_FN" ] || [ -z "$SETUP_CONDA_FN" ]; then
    echo "  FAIL: could not extract the PATH persistence functions from studio/setup.sh"
    exit 1
fi

# $1 CONDA_PREFIX ("" for none). Reads back ~/.bashrc, which the function writes to
# because it exists; ~/.profile is created either way and is not what is asserted here.
run_setup() {
    WORK=$(mktemp -d)
    : > "$WORK/.bashrc"
    (
        eval "$HARNESS"
        eval "$SETUP_CONDA_FN"
        eval "$SETUP_HAS_DIR_FN"
        eval "$SETUP_FN"
        HOME="$WORK"
        _SETUP_LOGIN_PATH="/usr/bin"
        unset UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL CONDA_PREFIX CONDA_DEFAULT_ENV
        if [ -n "$1" ]; then eval "${2:-CONDA_PREFIX}=\"\$1\"; export ${2:-CONDA_PREFIX}"; fi
        _setup_persist_uv_path "$WORK/.local/bin"
    )
    SETUP_RC=$(cat "$WORK/.bashrc")
    SETUP_FISH=$(cat "$WORK/.config/fish/conf.d/unsloth.fish" 2>/dev/null)
    SETUP_DIR="$WORK/.local/bin"
    rm -rf "$WORK"
}

run_setup ""
assert_contains "setup.sh, no conda: prepends" "$SETUP_RC" "export PATH=\"$SETUP_DIR:\$PATH\""
assert_contains "setup.sh, no conda: fish prepends" "$SETUP_FISH" "fish_add_path '$SETUP_DIR'"

run_setup "myenv" CONDA_DEFAULT_ENV
assert_contains "setup.sh, CONDA_DEFAULT_ENV alone: appends" "$SETUP_RC" "export PATH=\"\$PATH:$SETUP_DIR\""

run_setup "/opt/anaconda3"
assert_contains "setup.sh, active conda: appends" "$SETUP_RC" "export PATH=\"\$PATH:$SETUP_DIR\""
assert_not_contains "setup.sh, active conda: does not prepend" "$SETUP_RC" "export PATH=\"$SETUP_DIR:\$PATH\""
assert_contains "setup.sh, active conda: fish appends" "$SETUP_FISH" "fish_add_path -a '$SETUP_DIR'"

summary
