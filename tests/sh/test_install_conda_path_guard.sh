#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# The POSIX half of #5871. install.sh persists a PATH entry into a shell rc file (or
# ~/.config/fish/conf.d/unsloth.fish); written as a PREPEND it lands after conda's own init
# block, so every later shell resolves out of our directory first, an ordering that outlives
# the activation. The Windows half is tests/python/test_installer_conda_path_guard.py.
#
# Checked, in order:
#
#   1. Outside conda the written line is unchanged.
#   2. Inside conda the same directory is registered at the BACK.
#   3. Either spelling counts as present, so no machine collects two lines.
#   4. The advice printed when the rc file cannot be written matches what would have been
#      written, or the manual fix reintroduces the defect by hand.
#
# studio/setup.sh carries a third copy and is covered at the end: it runs in its own process
# right after install.sh. The functions are extracted from the installers and run rather
# than restated, so these cases track them instead of a copy.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_SH="$REPO_ROOT/install.sh"

echo "=== test_install_conda_path_guard ==="

# Both persisters plus the regex they share. Asserted non-empty below, so a rename is loud.
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
# The predicate both writers ask, extracted rather than restated so a change reaches here.
CONDA_FN=$(extract_function _unsloth_conda_env_active)
# The migration helper both writers call. Without it in the harness every repoint call is a
# `command not found` inside a conditional and the cases below still pass.
REPOINT_FN=$(extract_function _unsloth_repoint_rc_line)

for _chunk_label in _PATH_LINE_RE _persist_login_path_dir _persist_fish_path_dir _unsloth_conda_env_active _unsloth_repoint_rc_line; do
    case "$_chunk_label" in
        _PATH_LINE_RE)              _chunk="$PATH_LINE_RE_SRC" ;;
        _persist_login_path_dir)    _chunk="$LOGIN_FN" ;;
        _unsloth_conda_env_active)  _chunk="$CONDA_FN" ;;
        _unsloth_repoint_rc_line)   _chunk="$REPOINT_FN" ;;
        *)                          _chunk="$FISH_FN" ;;
    esac
    if [ -z "$_chunk" ]; then
        echo "  FAIL: could not extract $_chunk_label from install.sh"
        exit 1
    fi
done

# `step` and `substep` print in install.sh; captured so assertion 4 can read the advice.
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
        eval "$REPOINT_FN"
        eval "$LOGIN_FN"
        HOME="$WORK"
        SHELL="${4:-/bin/bash}"
        unset CONDA_PREFIX CONDA_DEFAULT_ENV
        # $3 names WHICH variable carries the activation: a hook exporting only
        # CONDA_DEFAULT_ENV still leaves the caller inside conda's ordering.
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

# Seeded WITH the marker the installer writes: that comment is the ownership record.
run_login '# Added by Unsloth installer
export PATH="$HOME/.local/bin:$PATH"' "/opt/anaconda3"
assert_eq "prepend already there: exactly one PATH line" "1" "$(printf '%s\n' "$RC_CONTENTS" | grep -c 'local/bin')"
# And that one line is the APPEND: counting lines alone passes whether the stale prepend was
# repointed or merely accepted as present, which is #5871.
assert_contains "stale prepend inside conda: rewritten as the append" "$RC_CONTENTS" \
    'export PATH="$PATH:$HOME/.local/bin"'
assert_not_contains "stale prepend inside conda: the prepend is gone" "$RC_CONTENTS" \
    'export PATH="$HOME/.local/bin:$PATH"'

# An identical line the USER wrote, with no marker above it, is not ours to move: rewriting
# one would push every executable in that directory behind the rest of PATH for good.
run_login 'export PATH="$HOME/.local/bin:$PATH"' "/opt/anaconda3"
assert_contains "unmarked user line inside conda: left alone" "$RC_CONTENTS" \
    'export PATH="$HOME/.local/bin:$PATH"'
assert_eq "unmarked user line inside conda: no second line added" "1" \
    "$(printf '%s\n' "$RC_CONTENTS" | grep -c 'local/bin')"

# The same stale prepend OUTSIDE conda is left alone: reordering it would change PATH for a
# user who never had the problem.
run_login '# Added by Unsloth installer
export PATH="$HOME/.local/bin:$PATH"' ""
assert_contains "stale prepend outside conda: left alone" "$RC_CONTENTS" \
    'export PATH="$HOME/.local/bin:$PATH"'
assert_eq "stale prepend outside conda: still one PATH line" "1" \
    "$(printf '%s\n' "$RC_CONTENTS" | grep -c 'local/bin')"

# 4. An unwritable rc file: the advice has to be the conda-safe line, not the prepend.
#    Chmod, not a directory: the failure path under test is the append redirect failing.
WORK=$(mktemp -d)
: > "$WORK/.bashrc"
chmod 400 "$WORK/.bashrc"
ADVICE=$(
    eval "$HARNESS"
    eval "$PATH_LINE_RE_SRC"
    eval "$CONDA_FN"
    eval "$REPOINT_FN"
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

# 5. CONDA_DEFAULT_ENV alone, which a hook can export without CONDA_PREFIX. install.ps1 has
#    always read both; the POSIX writers read only CONDA_PREFIX, so the same user got a
#    prepend on Linux and an append on Windows.
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
# fish reads none of the POSIX rc files: its own writer, and fish_add_path prepends.
run_fish() {
    WORK=$(mktemp -d)
    mkdir -p "$WORK/.config/fish/conf.d"
    printf '%s' "$1" > "$WORK/.config/fish/conf.d/unsloth.fish"
    (
        eval "$HARNESS"
        eval "$CONDA_FN"
        eval "$REPOINT_FN"
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

# -a ALONE is not an append to PATH: without --path it edits $fish_user_paths, which fish
# prepends to PATH. The assertions are exact so a regression to the bare spelling fails here.
run_fish "" "/opt/anaconda3"
assert_contains "fish, active conda: appends to PATH itself" "$FISH_CONTENTS" \
    "fish_add_path -a -P -m '$FISH_DIR'"

run_fish "" "myenv" CONDA_DEFAULT_ENV
assert_contains "fish, CONDA_DEFAULT_ENV alone: appends to PATH itself" "$FISH_CONTENTS" \
    "fish_add_path -a -P -m '$FISH_DIR'"

# A bare -a would leave the directory inside fish_user_paths, ahead of the active conda
# environment: the exact defect #5871 reports, arriving through fish instead of an rc file.
run_fish "" "/opt/anaconda3"
assert_eq "fish, active conda: no bare -a line" "0" \
    "$(printf '%s\n' "$FISH_CONTENTS" | grep -cxF "fish_add_path -a '$FISH_DIR'")"

# Both stale spellings are rewritten rather than accepted by the presence check. Seeded here
# rather than through run_fish because the line has to name the throwaway HOME. The -a -P
# spelling without -m is in the list too: fish leaves a directory already in
# $fish_user_paths where it is unless --move is given, so that line appended nothing.
for _stale in "fish_add_path '%s/.local/bin'" "fish_add_path -a '%s/.local/bin'" \
              "fish_add_path -a -P '%s/.local/bin'"; do
    WORK=$(mktemp -d)
    mkdir -p "$WORK/.config/fish/conf.d"
    # shellcheck disable=SC2059
    {
        echo "# Added by Unsloth installer"
        # shellcheck disable=SC2059
        printf "$_stale\n" "$WORK"
    } > "$WORK/.config/fish/conf.d/unsloth.fish"
    _stale_line=$(printf "$_stale" "$WORK")
    (
        eval "$HARNESS"
        eval "$CONDA_FN"
        eval "$REPOINT_FN"
        eval "$FISH_FN"
        HOME="$WORK"
        CONDA_PREFIX="/opt/anaconda3"; export CONDA_PREFIX
        unset CONDA_DEFAULT_ENV
        _persist_fish_path_dir "$WORK/.local/bin" "~/.local/bin"
    ) >/dev/null
    _after=$(cat "$WORK/.config/fish/conf.d/unsloth.fish")
    assert_contains "fish: stale [$_stale_line] rewritten to -a -P -m" "$_after" \
        "fish_add_path -a -P -m '$WORK/.local/bin'"
    assert_eq "fish: stale [$_stale_line] is gone" "0" \
        "$(printf '%s\n' "$_after" | grep -cxF "$_stale_line")"
    assert_eq "fish: stale [$_stale_line] leaves one line" "1" \
        "$(printf '%s\n' "$_after" | grep -c 'fish_add_path')"
    rm -rf "$WORK"
done

# Idempotence across both spellings, the same way as the POSIX arm.
WORK=$(mktemp -d)
mkdir -p "$WORK/.config/fish/conf.d"
printf "fish_add_path -a '%s/.local/bin'\n" "$WORK" > "$WORK/.config/fish/conf.d/unsloth.fish"
(
    eval "$HARNESS"
    eval "$CONDA_FN"
    eval "$REPOINT_FN"
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
# setup.sh carries its own copy of the migration helper, so it is extracted from setup.sh
# rather than reused from install.sh: the two copies have to be able to drift apart and
# still be checked.
SETUP_REPOINT_FN=$(extract_setup_function _unsloth_repoint_rc_line)
if [ -z "$SETUP_FN" ] || [ -z "$SETUP_HAS_DIR_FN" ] || [ -z "$SETUP_CONDA_FN" ] \
    || [ -z "$SETUP_REPOINT_FN" ]; then
    echo "  FAIL: could not extract the PATH persistence functions from studio/setup.sh"
    exit 1
fi

# $1 CONDA_PREFIX ("" for none). Reads back ~/.bashrc, which the function writes to
# because it exists; ~/.profile is created either way and is not what is asserted here.
# $SETUP_RC_SEED, when set, is a printf format given the throwaway HOME: that is how a
# stale line written by an earlier run can name a directory the caller does not know yet.
run_setup() {
    WORK=$(mktemp -d)
    if [ -n "${SETUP_RC_SEED:-}" ]; then
        # shellcheck disable=SC2059
        {
            echo "# Added by Unsloth setup"
            # shellcheck disable=SC2059
            printf "$SETUP_RC_SEED\n" "$WORK"
        } > "$WORK/.bashrc"
    else
        : > "$WORK/.bashrc"
    fi
    (
        eval "$HARNESS"
        eval "$SETUP_CONDA_FN"
        eval "$SETUP_REPOINT_FN"
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
assert_contains "setup.sh, active conda: fish appends to PATH itself" "$SETUP_FISH" \
    "fish_add_path -a -P -m '$SETUP_DIR'"
assert_eq "setup.sh, active conda: no bare -a line" "0" \
    "$(printf '%s\n' "$SETUP_FISH" | grep -cxF "fish_add_path -a '$SETUP_DIR'")"

# setup.sh's own migration pass. A prepend an earlier run left behind satisfies its
# presence check, so without the repoint the directory stays ahead of the active conda
# environment for ever. Both spellings it can have written are covered: the literal path
# and the $HOME-relative one.
# Set and unset around each call rather than as a command prefix: bash keeps an assignment
# that prefixes a FUNCTION call in the environment afterwards, which would seed every later
# case too.
SETUP_RC_SEED='export PATH="%s/.local/bin:$PATH"'
run_setup "/opt/anaconda3"
assert_contains "setup.sh, stale literal prepend inside conda: rewritten as the append" \
    "$SETUP_RC" "export PATH=\"\$PATH:$SETUP_DIR\""
assert_not_contains "setup.sh, stale literal prepend inside conda: the prepend is gone" \
    "$SETUP_RC" "export PATH=\"$SETUP_DIR:\$PATH\""
assert_eq "setup.sh, stale literal prepend inside conda: still one PATH line" "1" \
    "$(printf '%s\n' "$SETUP_RC" | grep -c 'local/bin')"

SETUP_RC_SEED='export PATH="$HOME/.local/bin:$PATH"'
run_setup "/opt/anaconda3"
assert_contains "setup.sh, stale home-relative prepend inside conda: rewritten as the append" \
    "$SETUP_RC" "export PATH=\"\$PATH:$SETUP_DIR\""
assert_not_contains "setup.sh, stale home-relative prepend inside conda: the prepend is gone" \
    "$SETUP_RC" 'export PATH="$HOME/.local/bin:$PATH"'
# The presence check reads the EXPANDED directory, so it cannot see the $HOME spelling.
# Gating the rewrite on it left the prepend alone and appended a second line under it.
assert_eq "setup.sh, stale home-relative prepend inside conda: still one PATH line" "1" \
    "$(printf '%s\n' "$SETUP_RC" | grep -c 'local/bin')"

# Outside conda the same stale prepend is the line this installer would write, so it is
# left exactly where it is.
SETUP_RC_SEED='export PATH="%s/.local/bin:$PATH"'
run_setup ""
assert_contains "setup.sh, stale prepend outside conda: left alone" "$SETUP_RC" \
    "export PATH=\"$SETUP_DIR:\$PATH\""
unset SETUP_RC_SEED

# ── rc-file permissions ────────────────────────────────────────────────────────────
# The rewrite renames a staged copy onto the user's profile, so whatever mode that copy
# has becomes the profile's mode for good. A plain `cp` does not carry the source's bits
# across: POSIX creates the destination with the source's mode as the mode ARGUMENT, and
# a mode argument is always modified by the file creation mask, so under `umask 077` a
# 0644 .bashrc came back 0600 -- or, in the other direction, a private file left more
# open than the user made it. Both copies of the helper are checked, because they are
# maintained separately and can drift.
check_mode_preserved() {
    _cmp_label="$1"
    _cmp_fn="$2"
    _cmp_mode="$3"
    _cmp_umask="$4"
    WORK=$(mktemp -d)
    {
        echo "# Added by Unsloth installer"
        echo "OLD LINE"
    } > "$WORK/.bashrc"
    chmod "$_cmp_mode" "$WORK/.bashrc"
    (
        eval "$_cmp_fn"
        umask "$_cmp_umask"
        _unsloth_repoint_rc_line "$WORK/.bashrc" "OLD LINE" "NEW LINE"
    )
    assert_eq "$_cmp_label: mode $_cmp_mode survives umask $_cmp_umask" "$_cmp_mode" \
        "$(ls -l "$WORK/.bashrc" | awk '{print $1}' | \
            awk '{ m = 0
                   for (i = 2; i <= 10; i++) {
                       c = substr($0, i, 1)
                       if (c != "-") m += (c == "r" ? 4 : (c == "w" ? 2 : 1)) * \
                           (i <= 4 ? 100 : (i <= 7 ? 10 : 1))
                   }
                   printf "%d", m }')"
    assert_eq "$_cmp_label: the line was rewritten" "1" \
        "$(grep -cxF "NEW LINE" "$WORK/.bashrc")"
    rm -rf "$WORK"
}

check_mode_preserved "install.sh" "$REPOINT_FN" 644 077
check_mode_preserved "install.sh" "$REPOINT_FN" 600 022
check_mode_preserved "studio/setup.sh" "$SETUP_REPOINT_FN" 644 077
check_mode_preserved "studio/setup.sh" "$SETUP_REPOINT_FN" 600 022

summary
