#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The POSIX half of #5871: a PATH entry persisted as a PREPEND lands after conda's init block,
# so inside conda it must be appended. Functions are extracted from install.sh and setup.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INSTALL_SH="$REPO_ROOT/install.sh"

echo "=== test_install_conda_path_guard ==="

# Asserted non-empty below, so a rename is loud.
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
CONDA_FN=$(extract_function _unsloth_conda_env_active)
# Without it every repoint call is a `command not found` inside a conditional and cases still pass.
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

HARNESS=$(cat <<'EOF'
step()    { echo "STEP $1 $2"; }
substep() { echo "SUBSTEP $1"; }
C_WARN=""
EOF
)

# $1 rc file contents ("" for empty), $2 CONDA_PREFIX ("" for none).
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
        # $3 names which variable carries the activation: CONDA_DEFAULT_ENV alone still counts.
        if [ -n "$2" ]; then eval "${3:-CONDA_PREFIX}=\"\$2\"; export ${3:-CONDA_PREFIX}"; fi
        _persist_login_path_dir "$WORK/.local/bin" '$HOME/.local/bin' "~/.local/bin" '\.local/bin' "${RC_FILE:-$WORK/.bashrc}"
    )
    RC_CONTENTS=$(cat "${RC_FILE:-$WORK/.bashrc}")
    rm -rf "$WORK"
}

run_login "" ""
assert_contains "no conda: writes a PATH prepend" "$RC_CONTENTS" 'export PATH="$HOME/.local/bin:$PATH"'

run_login "" "/opt/anaconda3"
assert_contains "active conda: writes a PATH append" "$RC_CONTENTS" 'export PATH="$PATH:$HOME/.local/bin"'
assert_not_contains "active conda: does not prepend" "$RC_CONTENTS" 'export PATH="$HOME/.local/bin:$PATH"'

# Either spelling is already-present, both directions.
run_login 'export PATH="$PATH:$HOME/.local/bin"' ""
assert_not_contains "append already there: no prepend added" "$RC_CONTENTS" 'export PATH="$HOME/.local/bin:$PATH"'
assert_eq "append already there: exactly one PATH line" "1" "$(printf '%s\n' "$RC_CONTENTS" | grep -c 'local/bin')"

# Seeded with the installer's marker: that comment is the ownership record.
run_login '# Added by Unsloth installer
export PATH="$HOME/.local/bin:$PATH"' "/opt/anaconda3"
assert_eq "prepend already there: exactly one PATH line" "1" "$(printf '%s\n' "$RC_CONTENTS" | grep -c 'local/bin')"
# Counting lines alone would pass whether the stale prepend was repointed or merely accepted.
assert_contains "stale prepend inside conda: rewritten as the append" "$RC_CONTENTS" \
    'export PATH="$PATH:$HOME/.local/bin"'
assert_not_contains "stale prepend inside conda: the prepend is gone" "$RC_CONTENTS" \
    'export PATH="$HOME/.local/bin:$PATH"'

# An identical line the user wrote (no marker) is not ours to move.
run_login 'export PATH="$HOME/.local/bin:$PATH"' "/opt/anaconda3"
assert_contains "unmarked user line inside conda: left alone" "$RC_CONTENTS" \
    'export PATH="$HOME/.local/bin:$PATH"'
assert_eq "unmarked user line inside conda: no second line added" "1" \
    "$(printf '%s\n' "$RC_CONTENTS" | grep -c 'local/bin')"

run_login '# Added by Unsloth installer
export PATH="$HOME/.local/bin:$PATH"' ""
assert_contains "stale prepend outside conda: left alone" "$RC_CONTENTS" \
    'export PATH="$HOME/.local/bin:$PATH"'
assert_eq "stale prepend outside conda: still one PATH line" "1" \
    "$(printf '%s\n' "$RC_CONTENTS" | grep -c 'local/bin')"

# Unwritable rc: the advice must be the conda-safe line. Chmod, so the append redirect fails.
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
    # Root can write a mode-400 file; skip loudly.
    ok "unwritable rc: skipped (this user can write a mode-400 file)"
fi
rm -rf "$WORK"

# CONDA_DEFAULT_ENV alone, which a hook can export without CONDA_PREFIX.
run_login "" "myenv" CONDA_DEFAULT_ENV
assert_contains "CONDA_DEFAULT_ENV alone: writes a PATH append" "$RC_CONTENTS" 'export PATH="$PATH:$HOME/.local/bin"'
assert_not_contains "CONDA_DEFAULT_ENV alone: does not prepend" "$RC_CONTENTS" 'export PATH="$HOME/.local/bin:$PATH"'

RC_NAME=.zshrc run_login "" "/opt/anaconda3" CONDA_PREFIX /bin/zsh
assert_contains "zsh rc, active conda: appends" "$RC_CONTENTS" 'export PATH="$PATH:$HOME/.local/bin"'
RC_NAME=.zshrc run_login "" "" CONDA_PREFIX /bin/zsh
assert_contains "zsh rc, no conda: prepends" "$RC_CONTENTS" 'export PATH="$HOME/.local/bin:$PATH"'

# fish reads none of the POSIX rc files, and fish_add_path prepends.
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

# -a alone edits $fish_user_paths, which fish prepends to PATH; --path is required.
run_fish "" "/opt/anaconda3"
assert_contains "fish, active conda: appends to PATH itself" "$FISH_CONTENTS" \
    "fish_add_path -a -P -m '$FISH_DIR'"

run_fish "" "myenv" CONDA_DEFAULT_ENV
assert_contains "fish, CONDA_DEFAULT_ENV alone: appends to PATH itself" "$FISH_CONTENTS" \
    "fish_add_path -a -P -m '$FISH_DIR'"

run_fish "" "/opt/anaconda3"
assert_eq "fish, active conda: no bare -a line" "0" \
    "$(printf '%s\n' "$FISH_CONTENTS" | grep -cxF "fish_add_path -a '$FISH_DIR'")"

# Stale spellings are rewritten, including -a -P without -m: fish keeps an existing
# $fish_user_paths entry in place unless --move is given.
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

# studio/setup.sh persists the uv dir into the same rc files right after install.sh.
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
# Extracted from setup.sh, not install.sh: the two copies can drift.
SETUP_REPOINT_FN=$(extract_setup_function _unsloth_repoint_rc_line)
if [ -z "$SETUP_FN" ] || [ -z "$SETUP_HAS_DIR_FN" ] || [ -z "$SETUP_CONDA_FN" ] \
    || [ -z "$SETUP_REPOINT_FN" ]; then
    echo "  FAIL: could not extract the PATH persistence functions from studio/setup.sh"
    exit 1
fi

# $1 CONDA_PREFIX. $SETUP_RC_SEED, when set, is a printf format given the throwaway HOME.
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

# Set and unset around each call: bash keeps an assignment prefixing a FUNCTION call in the
# environment afterwards.
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
# The presence check reads the expanded dir, so it cannot see the $HOME spelling.
assert_eq "setup.sh, stale home-relative prepend inside conda: still one PATH line" "1" \
    "$(printf '%s\n' "$SETUP_RC" | grep -c 'local/bin')"

SETUP_RC_SEED='export PATH="%s/.local/bin:$PATH"'
run_setup ""
assert_contains "setup.sh, stale prepend outside conda: left alone" "$SETUP_RC" \
    "export PATH=\"$SETUP_DIR:\$PATH\""
unset SETUP_RC_SEED

# The rewrite renames a staged copy onto the profile; `cp` applies the umask, so the mode
# must be preserved explicitly. Both helper copies are checked.
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
