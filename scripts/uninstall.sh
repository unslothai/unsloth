#!/usr/bin/env sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Unsloth Studio uninstaller (macOS / Linux / WSL). Run --help for details.
# Custom roots (UNSLOTH_STUDIO_HOME / STUDIO_HOME) come from studio.conf.
#
# Usage: curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh | sh

set -e

_usage() {
    cat <<'EOF'
Unsloth Studio uninstaller (macOS / Linux / WSL).

Usage:
  curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh | sh
  sh scripts/uninstall.sh

To read this help from the piped form, sh needs -s so the arguments reach the
script instead of the shell. Spelled out, with the same URL as above:
  curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh | sh -s -- --help

Never pipe to `sh -h` expecting help. Neither form prints this message: where
-h is accepted (bash, zsh, macOS /bin/sh) it is the shell's own hashall option,
so the shell consumes it and the script uninstalls with no arguments; where it
is not (dash, busybox sh) the shell exits with "Illegal option -h".

Stops running Unsloth Studio servers, then removes the install dir, launcher
data dir, CLI shim, desktop shortcut, macOS .app bundle and Launch Services
entry. In a default-mode install it also removes the shared prebuilts that sit
beside the install dir: ~/.unsloth/{llama.cpp,node,whisper.cpp,.cache}. The
Hugging Face cache at ~/.cache/huggingface is left in place, as is anything
else you keep under ~/.unsloth.

On WSL it also removes this distro's Windows-side shortcuts under /mnt/*/Users,
strips the Unsloth block from ~/.bashrc, and uses sudo to delete
/etc/profile.d/unsloth-rocm-wsl.sh.

Options:
  -h, --help  Print this message and exit without removing anything.

Run with no arguments to uninstall. Unrecognized arguments never trigger
removal.

Environment:
  UNSLOTH_STUDIO_HOME       Also remove this custom install root. Pass the value
                            used at install time.
  STUDIO_HOME               Alias for the above, ignored when both are set.
  UNSLOTH_UNINSTALL_ROCM=1  Also remove system ROCm (WSL only). Off by default
                            because ROCm is a shared prerequisite.
EOF
}

# Stop an Unsloth server via its PID file (written by install.sh's _spawn_terminal).
_kill_pid_file() {
    _pid_file="$1"
    [ -f "$_pid_file" ] || return 0
    _pid=$(sed -n '1s/[^0-9].*//p' "$_pid_file" 2>/dev/null || true)
    if [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null; then
        kill -TERM "$_pid" 2>/dev/null || true
        # Wait up to 10s for graceful shutdown.
        _i=0
        while kill -0 "$_pid" 2>/dev/null && [ "$_i" -lt 20 ]; do
            sleep 0.5
            _i=$((_i + 1))
        done
        kill -0 "$_pid" 2>/dev/null && kill -KILL "$_pid" 2>/dev/null || true
    fi
    rm -f "$_pid_file" 2>/dev/null || true
}

# BRE-escape a path so it can be embedded in a pkill -f regex.
_pkill_escape() {
    printf '%s' "$1" | sed -e 's:[][\\.^$*+?{|}()/]:\\&:g'
}

# sd.cpp roots whose sd-server has to be stopped: the default $HOME/.unsloth/stable-diffusion.cpp,
# plus per custom root the nested <root>/stable-diffusion.cpp and the legacy <parent> sibling.
# A resident sd-server survives unlinking its binary, and the custom root is removed wholesale
# below, so without the nested path the tree goes and the server keeps running.
# The owner marker gates only the paths that SURVIVE when unowned (the default and the sibling), so
# an unrelated checkout keeps its server; the nested path of a root this run deletes is not gated.
_owned_sd_cpp_roots() {
    _default_sd="$HOME/.unsloth/stable-diffusion.cpp"
    [ -f "$_default_sd/.unsloth-studio-owned" ] && printf '%s\n' "$_default_sd"
    _custom_studio_roots 2>/dev/null | while IFS= read -r _root; do
        [ -n "$_root" ] || continue
        _sd_root="$_root/stable-diffusion.cpp"
        if [ -f "$_sd_root/.unsloth-studio-owned" ] || _is_studio_root "$_root"; then
            [ -d "$_sd_root" ] && printf '%s\n' "$_sd_root"
        fi
    done
    _sd_cpp_sibling_bases 2>/dev/null | while IFS= read -r _root; do
        [ -n "$_root" ] || continue
        _sd_root="$(dirname "$_root")/stable-diffusion.cpp"
        [ -f "$_sd_root/.unsloth-studio-owned" ] && printf '%s\n' "$_sd_root"
    done
}

# Every root an older build could have hung its sd.cpp sibling off: the canonicalized custom roots
# and the lexical ones. They differ only when the Unsloth home is itself a symlink, and there only
# the lexical form (what the old plain `dirname` produced) finds the tree. Every use is gated on
# the owner marker, which is what keeps an unrelated checkout at either path safe.
_sd_cpp_sibling_bases() {
    {
        _custom_studio_roots 2>/dev/null
        _custom_studio_roots lexical 2>/dev/null
    } | awk '!seen[$0]++'
}

# pkill resident sd-server / sd-cli under an owned sd.cpp root before that tree is removed (a live
# native server keeps running after its binary is unlinked). Anchored on the owned root.
_stop_owned_sd_cpp_processes() {
    _signal="$1"
    command -v pkill >/dev/null 2>&1 || return 0
    _owned_sd_cpp_roots | while IFS= read -r _root; do
        [ -n "$_root" ] || continue
        [ -d "$_root" ] || continue
        _re=$(_pkill_escape "$_root")
        pkill "-$_signal" -f "^${_re}/([^ ]*/)?sd-(server|cli)( |\$)" 2>/dev/null || true
    done
}

# Numeric owner of $HOME, empty if unresolvable. Not always the caller: macOS sudo keeps
# HOME (env_keep), so the home stays the invoking user's while euid is 0.
_home_uid() {
    # -L: stat lstats by default, so a root-owned link to a user home would read as uid 0.
    _hu=$(stat -L -c %u "$HOME" 2>/dev/null || stat -L -f %u "$HOME" 2>/dev/null || true)
    case "$_hu" in ''|*[!0-9]*) _hu=$(id -u 2>/dev/null || true) ;; esac
    case "$_hu" in *[!0-9]*) _hu= ;; esac
    printf '%s\n' "$_hu"
}

# Run "$@" as the $HOME owner when we are root and it is someone else, so per-user daemons
# see the right domain. Plain "$@" otherwise, which is every non-elevated run.
_run_as_home_owner() {
    _ro=$(_home_uid)
    if [ "$(id -u 2>/dev/null || echo 0)" = "0" ] && [ -n "$_ro" ] && [ "$_ro" != "0" ] &&
       command -v launchctl >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
        launchctl asuser "$_ro" sudo -u "#$_ro" "$@"
    else
        "$@"
    fi
}

# Is the desktop app running? Only reached when there is no pkill to ask, so read /proc.
# Scoped to the owner of the $HOME being cleared, matching the pkill -u call below.
_studio_app_running() {
    [ -d /proc ] || return 1
    _sar_uid=$(_home_uid)
    for _sar_p in /proc/[0-9]*; do
        [ -r "$_sar_p/comm" ] || continue
        _sar_comm=$(cat "$_sar_p/comm" 2>/dev/null || true)
        [ "$_sar_comm" = "unsloth-studio" ] || continue
        if [ -n "$_sar_uid" ]; then
            _sar_owner=$(stat -c %u "$_sar_p" 2>/dev/null || stat -f %u "$_sar_p" 2>/dev/null || true)
            [ "$_sar_owner" = "$_sar_uid" ] || continue
        fi
        return 0
    done
    return 1
}

_pkill_studio() {
    # Prefer PID files written by _spawn_terminal so we only touch our own installs.
    for _data_dir in "$HOME/.local/share/unsloth" $(_custom_studio_data_dirs); do
        [ -d "$_data_dir" ] || continue
        for _pf in "$_data_dir"/studio-*.pid; do
            [ -f "$_pf" ] && _kill_pid_file "$_pf"
        done
    done

    if ! command -v pkill >/dev/null 2>&1; then
        # No procps (install.sh never requires it): the PID sweep above is all we have. A live
        # app re-creates the profile right after the delete, so do not claim a clean removal.
        if _studio_app_running; then
            echo "  pkill not found and Unsloth Studio is running; close it and re-run" >&2
            _set_marker "$_REMOVE_FAILED_FLAG"
        fi
        return 0
    fi

    # Scope fallback patterns to the install roots we are removing so a
    # different Unsloth install (different UNSLOTH_STUDIO_HOME) is not touched.
    _kill_roots="$HOME/.unsloth/studio"
    _roots_from_conf=$(_custom_studio_roots 2>/dev/null || true)
    [ -n "$_roots_from_conf" ] && _kill_roots="$_kill_roots
$_roots_from_conf"

    printf '%s\n' "$_kill_roots" | while IFS= read -r _root; do
        [ -n "$_root" ] || continue
        [ -d "$_root" ] || continue
        _re=$(_pkill_escape "$_root")
        # `unsloth studio` default-port, `-p N` and `--port N` forms, anchored on the venv path.
        for _pat in \
            "${_re}/unsloth_studio/bin/[^ ]* studio( |\$|.*-p[ =][0-9])" \
            "${_re}/unsloth_studio/bin/[^ ]* studio.*--port[ =][0-9]" \
            "${_re}/.*studio/backend/run\.py"
        do
            pkill -TERM -f "$_pat" 2>/dev/null || true
        done
    done
    sleep 0.5
    printf '%s\n' "$_kill_roots" | while IFS= read -r _root; do
        [ -n "$_root" ] || continue
        [ -d "$_root" ] || continue
        _re=$(_pkill_escape "$_root")
        for _pat in \
            "${_re}/unsloth_studio/bin/[^ ]* studio( |\$|.*-p[ =][0-9])" \
            "${_re}/unsloth_studio/bin/[^ ]* studio.*--port[ =][0-9]" \
            "${_re}/.*studio/backend/run\.py"
        do
            pkill -KILL -f "$_pat" 2>/dev/null || true
        done
    done

    # sd-server / sd-cli survive unlinking their binary, so stop them before their trees go.
    _stop_owned_sd_cpp_processes TERM
    sleep 0.5
    _stop_owned_sd_cpp_processes KILL

    # The app's WebView helpers re-create the caches removed below, so it has to die here. -x is
    # exact, so the "unsloth" CLI shim never matches. -u takes the owner of the $HOME being
    # cleared, not the caller (macOS sudo keeps HOME), so a root run spares other users and an
    # unknown owner skips entirely. Numeric uid and signal-first for BSD pkill (signal in argv[1]).
    _studio_uid=$(_home_uid)
    if [ -n "$_studio_uid" ]; then
        pkill -TERM -x -u "$_studio_uid" unsloth-studio 2>/dev/null || true
        sleep 0.5
        pkill -KILL -x -u "$_studio_uid" unsloth-studio 2>/dev/null || true
    fi
}

# Summary state in files, not variables: custom roots are removed inside a pipeline subshell,
# where an assignment would never reach the summary.
#   remove-failed  an rm failed, or a root was skipped while still holding data
#   db-removed     a removed install root actually held studio.db
#   db-kept        the default root was REFUSED and still holds studio.db. Its own flag, not
#                  remove-failed: nothing failed, so "remove those paths by hand" is wrong here.
# studio.db holds the chat history (backend/storage/studio_db.py), not the provider API keys:
# providers_db.py keeps those in the browser's localStorage only. An env-mode install keeps it in
# a custom root a bare run cannot discover, so claim the history is gone only if one was deleted.
# mktemp -d, not a $TMPDIR name: private (0700) and unpredictable, so nothing else can pre-create.
_MARKER_DIR=$(mktemp -d 2>/dev/null || true)
_REMOVE_FAILED_FLAG=""
_DB_REMOVED_FLAG=""
_DB_KEPT_FLAG=""
if [ -n "$_MARKER_DIR" ] && [ -d "$_MARKER_DIR" ]; then
    _REMOVE_FAILED_FLAG="$_MARKER_DIR/remove-failed"
    _DB_REMOVED_FLAG="$_MARKER_DIR/db-removed"
    _DB_KEPT_FLAG="$_MARKER_DIR/db-kept"
fi

# `printf`, never `: > "$f"`: `:` is a POSIX special builtin, so a redirection error on it kills a
# non-interactive shell outright (dash, busybox ash) and `|| true` does not stop it.
_set_marker() {
    [ -n "$1" ] || return 0
    printf '' > "$1" 2>/dev/null || true
    return 0
}
_marker_set() { [ -n "$1" ] && [ -f "$1" ]; }
# No marker storage means no record of what failed, so the summary must not claim success.
# Re-checked, not trusted from startup: the dir can vanish or lose write access mid-run, after
# which _set_marker silently drops every failure.
_markers_unavailable() {
    [ -n "$_MARKER_DIR" ] || return 0
    [ -d "$_MARKER_DIR" ] || return 0
    [ -w "$_MARKER_DIR" ] || return 0
    return 1
}

# EXIT, not a line at the end of main: --help, a bad argument and `set -e` all skip that.
_cleanup_markers() {
    if [ -n "$_MARKER_DIR" ]; then
        rm -rf "$_MARKER_DIR" 2>/dev/null || true
    fi
}
trap _cleanup_markers EXIT

# A removal that got part way can take the sentinels and then fail on a locked child, leaving a
# root the next run's gate would refuse. Put the marker back so a retry recognises it.
_restore_owner_marker() {
    [ -d "$1" ] || return 0
    if [ -e "$1/.unsloth-studio-owned" ] || [ -L "$1/.unsloth-studio-owned" ]; then return 0; fi
    printf '' > "$1/.unsloth-studio-owned" 2>/dev/null || true
    return 0
}

# Remove an install root and record whether its studio.db really went with it. The check runs on
# the RESOLVED path: a relocated install (~/.unsloth/studio a symlink to another disk) passes
# `-f "$root/studio.db"` through the link, but `rm -rf` unlinks only the link, after which the
# path reads as absent either way. Verifying rather than chasing the link is deliberate: following
# a symlink out of the expected location to `rm -rf` its target is what the deny lists prevent.
_remove_root_recording_db() {
    _rrd_root="$1"
    # shellcheck disable=SC1007
    _rrd_real=$(CDPATH= cd -P -- "$_rrd_root" 2>/dev/null && pwd -P) || _rrd_real=""
    [ -n "$_rrd_real" ] || _rrd_real="$_rrd_root"
    _rrd_had_db=0
    _rrd_db="$_rrd_real/studio.db"
    if [ -f "$_rrd_db" ]; then
        _rrd_had_db=1
        # The db itself can be a symlink out of the tree: -f follows it but the rm unlinks only
        # the link, so track where the bytes are. readlink without -f (BSD got it in macOS 12.3).
        if [ -L "$_rrd_db" ]; then
            _rrd_link=$(readlink "$_rrd_db" 2>/dev/null || true)
            if [ -n "$_rrd_link" ]; then
                case "$_rrd_link" in
                    /*) ;;
                    *) _rrd_link="$(dirname "$_rrd_db")/$_rrd_link" ;;
                esac
                # shellcheck disable=SC1007
                _rrd_ldir=$(CDPATH= cd -P -- "$(dirname "$_rrd_link")" 2>/dev/null && pwd -P) \
                    || _rrd_ldir=""
                [ -n "$_rrd_ldir" ] && _rrd_db="$_rrd_ldir/$(basename "$_rrd_link")"
            fi
        fi
    fi
    _remove_path "$_rrd_root"
    _restore_owner_marker "$_rrd_root"
    if [ "$_rrd_had_db" = 1 ]; then
        if [ -f "$_rrd_db" ]; then
            # Only the link went, or the delete failed: the data is still on disk.
            _set_marker "$_REMOVE_FAILED_FLAG"
        else
            _set_marker "$_DB_REMOVED_FLAG"
        fi
    fi
    return 0
}

_remove_path() {
    _p="$1"
    if [ -e "$_p" ] || [ -L "$_p" ]; then
        if rm -rf "$_p" 2>/dev/null; then
            echo "  removed: $_p"
        else
            echo "  could not remove: $_p" >&2
            # A marker file, not a variable: custom roots are removed in a pipeline subshell.
            _set_marker "$_REMOVE_FAILED_FLAG"
        fi
    fi
}

# An install lock, and only an install lock.
#
# prebuilt_core.install_lock creates these with os.open(O_CREAT | O_EXCL) and writes a pid, so
# the lock is always a regular file. _remove_path is rm -rf, which in a user-chosen master root
# would take a whole tree that merely happens to be named .node.install.lock -- a name the user
# owns as much as any other, since the root is theirs. Nothing outside these fixed names reaches
# here, so the test costs nothing and removes the one shape we never meant to delete.
#
# A symlink is unlinked rather than followed: -f is true for a link to a file, and rm on the link
# takes the link.
_remove_lock_file() {
    _rlf="$1"
    if [ -L "$_rlf" ] || [ -f "$_rlf" ]; then
        _remove_path "$_rlf"
    elif [ -e "$_rlf" ]; then
        echo "  keeping non-file at an install-lock path: $_rlf" >&2
    fi
}

# $1 override, $2 default. A relative override is invalid per XDG and dropped by dirs (which
# Tauri resolves through), so honouring one would spare the real data and rm -rf under our cwd.
_xdg_dir() {
    case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$2" ;; esac
}

# Accept as Unsloth root only if an Unsloth sentinel exists (matching install.sh's env-mode
# ownership guard). A bare unsloth_studio/ directory is NOT enough: require the install-time owner
# marker so a user directory that happens to contain such a folder is safe.
# Is $1 a Python venv? What the gate reads out of one is evidence only if the directory really
# is one; a bare file at that path is somebody else's.
# A sentinel this gate may trust. Deliberately just -f, following links: refusing one that is a
# link, or that sits inside a linked directory, buys nothing here and costs a supported install.
# It buys nothing because anyone who can plant a link at that path can plant a plain file there
# instead, which this has always accepted. It costs a supported install because relocating a
# multi-gigabyte venv with a symlink leaves a REAL marker behind a link, and refusing it strands
# the install, which is the failure this whole gate exists to prevent. The link test belongs in
# install.sh's _claim_sentinel, where following one would TRUNCATE the target rather than read it.
_is_owner_marker() { [ -f "$1" ]; }

# No -L here either, and for the same reason: a relocated venv is still a venv. The leftover scan
# below rejects links on its own, where the name came from a glob rather than from us.
_is_venv_dir() {
    [ -d "$1" ] || return 1
    [ -f "$1/pyvenv.cfg" ] && return 0
    [ -f "$1/bin/python" ] && return 0
    return 1
}

# The exact name an installer gives a moved-aside venv: <prefix>.<stamp>.<pid>[.<n>], with a
# 14-digit stamp or install.sh's "time" date(1) fallback. install.sh keeps every other spelling
# as the user's data (_studio_venv_rollback_must_be_preserved); do not be looser.
_is_installer_leftover_name() {
    _l=${1##*/}
    # Only the rollback name carries a collision counter (install.sh:823); .venv.invalid is
    # written once per run, so it takes no suffix.
    _l_suffix_ok=false
    case "$_l" in
        unsloth_studio.rollback.*) _l=${_l#unsloth_studio.rollback.}; _l_suffix_ok=true ;;
        .venv.invalid.*)           _l=${_l#.venv.invalid.} ;;
        *) return 1 ;;
    esac
    _l_rest=${_l#*.}
    [ "$_l_rest" != "$_l" ] || return 1
    _l_stamp=${_l%%.*}
    case "$_l_stamp" in
        time) ;;
        ''|*[!0-9]*) return 1 ;;
        *) [ "${#_l_stamp}" -eq 14 ] || return 1 ;;
    esac
    _l_pid=${_l_rest%%.*}
    case "$_l_pid" in ''|*[!0-9]*) return 1 ;; esac
    _l_suffix=${_l_rest#*.}
    if [ "$_l_suffix" != "$_l_rest" ]; then
        [ "$_l_suffix_ok" = true ] || return 1
        case "$_l_suffix" in ''|*[!0-9]*) return 1 ;; esac
    fi
    return 0
}

_is_studio_root() {
    _r="$1"
    # $2 = "managed": $_r is the default root install.sh manages, $HOME/.unsloth/studio.
    _managed="${2:-}"
    [ -n "$_r" ] || return 1
    # install.sh writes the first when it creates the root, before the uv cache and long before
    # the venv, so a partial install identifies itself instead of being guessed at. The last is
    # the legacy venv name, which only install.sh writes.
    _is_owner_marker "$_r/.unsloth-studio-owned" && return 0
    _is_owner_marker "$_r/share/studio.conf" && return 0
    _is_owner_marker "$_r/unsloth_studio/.unsloth-studio-owned" && return 0
    _is_owner_marker "$_r/.venv/.unsloth-studio-owned" && return 0
    if [ -L "$_r/bin/unsloth" ]; then
        _t=$(readlink "$_r/bin/unsloth" 2>/dev/null || true)
        case "$_t" in *unsloth_studio/bin/unsloth) return 0 ;; esac
    fi
    # All a pre-marker install has left is bin/unsloth inside the venv, pip's console script,
    # which ANY venv with the wheel has: proof only at the managed root, which install.sh:2987
    # also makes the only root that can hold the layout, or a stale UNSLOTH_STUDIO_HOME deletes
    # the project it points at.
    [ "$_managed" = managed ] || return 1
    for _v in unsloth_studio .venv; do
        _is_venv_dir "$_r/$_v" || continue
        [ -f "$_r/$_v/bin/unsloth" ] && return 0
    done
    # An install that died between moving the old venv aside (install.sh:3027, :819) and writing
    # the marker (install.sh:3190) leaves only these, and only install.sh makes either name,
    # always by renaming a venv.
    for _p in "$_r"/unsloth_studio.rollback.* "$_r"/.venv.invalid.*; do
        # Never a link: install.sh refuses to prune a rollback symlink either.
        [ -L "$_p" ] && continue
        _is_installer_leftover_name "$_p" || continue
        _is_venv_dir "$_p" && return 0
    done
    return 1
}

# Hard deny list: never delete /, $HOME, $HOME's parent, or system paths.
_is_unsafe_root() {
    _r="$1"
    [ -z "$_r" ] && return 0
    case "$_r" in /|""|"$HOME"|"$HOME/") return 0 ;; esac
    case "$_r" in /bin|/sbin|/etc|/usr|/usr/*|/var|/var/*|/opt|/opt/*|/Library|/Library/*|/System|/System/*|/Applications|/Applications/*) return 0 ;; esac
    _parent=$(dirname "$HOME" 2>/dev/null || echo "")
    [ -n "$_parent" ] && [ "$_r" = "$_parent" ] && return 0
    return 1
}

# Print share/ dirs of known custom roots (where PID files live).
_custom_studio_data_dirs() {
    _custom_studio_roots 2>/dev/null | while IFS= read -r _r; do
        [ -d "$_r/share" ] && printf '%s\n' "$_r/share"
    done
}

# Resolve a custom install root from any of:
#   1. UNSLOTH_STUDIO_HOME / STUDIO_HOME env vars at uninstall time
#   2. Default-mode studio.conf at $HOME/.local/share/unsloth/studio.conf
#   3. Env-mode studio.conf at $<root>/share/studio.conf (discovered via 1)
# install.sh writes UNSLOTH_EXE='<root>/unsloth_studio/bin/unsloth', so the install root is three
# dirnames up. Each discovered non-default root is printed on its own line, de-duplicated.
# The master root UNSLOTH_HOME names, or empty. studio/ is its child and llama.cpp, node and
# whisper.cpp are its other children, so removing the Studio root alone strands them. Stripped
# and tilde-expanded like storage_roots.unsloth_home() and studio/setup.sh, or a padded value
# would name a directory neither install nor uninstall agrees on.
_master_root() {
    _mr=$(printf '%s' "${UNSLOTH_HOME:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    # The note setup.sh leaves in the Studio tree, when this run has no UNSLOTH_HOME of its own.
    # `UNSLOTH_HOME=/mnt/portable unsloth studio update` installs the runtimes there and leaves
    # nothing in this environment, so without the note an uninstall later removed the Studio
    # tree and stranded them. Every Studio root this script already knows is consulted, and the
    # first readable note wins; the deny list and the marker gate below still apply to whatever
    # it names, so a stale note cannot license a removal the environment could not.
    if [ -z "$_mr" ]; then
        for _mr_conf in "$HOME/.unsloth/studio/share/.unsloth-master-root" \
                        "${UNSLOTH_STUDIO_HOME:-}/share/.unsloth-master-root" \
                        "${STUDIO_HOME:-}/share/.unsloth-master-root"; do
            case "$_mr_conf" in /share/*) continue ;; esac
            [ -f "$_mr_conf" ] || continue
            # One line, first only: a note that grew a second line is not one we wrote.
            _mr=$(head -n 1 "$_mr_conf" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//') \
                || _mr=""
            [ -n "$_mr" ] && break
        done
    fi
    [ -n "$_mr" ] || return 0
    # shellcheck disable=SC2088
    case "$_mr" in
        "~") _mr="$HOME" ;;
        "~/"*) _mr="$HOME/${_mr#'~/'}" ;;
    esac
    # shellcheck disable=SC1007
    # || _mr_canon="": a bare assignment takes the substitution's exit status, and this script
    # runs under set -e. A root that is already gone, or on a disconnected drive, is the ordinary
    # case here -- a second uninstall, a portable install on an unplugged disk -- and without the
    # guard it killed the whole run before any of the rest of the cleanup. Ordinary bash hides
    # this by clearing errexit inside command substitution; dash and `sh` in POSIX mode, which
    # is what the advertised `| sh` one-liner uses on Debian, Ubuntu and WSL, do not.
    # setup.sh's master-root block already guards the same call this way.
    _mr_canon=$(CDPATH= cd -P -- "$_mr" 2>/dev/null && pwd -P) || _mr_canon=""
    [ -n "$_mr_canon" ] && _mr="$_mr_canon"
    case "$_mr" in "$HOME/.unsloth"|/|"") return 0 ;; esac
    printf '%s\n' "$_mr"
}

_custom_studio_roots() {
    # $1 = "lexical": skip the canonicalization (see the legacy sd.cpp sibling below). Reset on
    # every call, so a plain call is never affected by a preceding lexical one.
    _studio_roots_lexical="${1:-}"
    _seen=""
    _emit() {
        _r="$1"
        [ -z "$_r" ] && return 0
        # Tilde expansion (env vars are not subject to it on quoted assignment), matching
        # install.sh's _resolve_studio_destinations. The literal "~/" pattern is intentional.
        # shellcheck disable=SC2088
        case "$_r" in
            "~") _r="$HOME" ;;
            "~/"*) _r="$HOME/${_r#'~/'}" ;;
        esac
        # Canonicalize so syntactic variants ($HOME/../$USER, trailing slash) resolve to the same
        # path and hit the _is_unsafe_root deny list. Skipped for the lexical pass, which rebuilds
        # the path an older build derived with a plain dirname (see _sd_cpp_sibling_bases).
        if [ "${_studio_roots_lexical:-}" != "lexical" ]; then
            # shellcheck disable=SC1007
            _canon=$(CDPATH= cd -P -- "$_r" 2>/dev/null && pwd -P)
            [ -n "$_canon" ] && _r="$_canon"
        fi
        case "$_r" in "$HOME/.unsloth/studio"|/|"") return 0 ;; esac
        case ":$_seen:" in *":$_r:"*) return 0 ;; esac
        _seen="$_seen:$_r"
        printf '%s\n' "$_r"
    }
    _from_conf() {
        [ -f "$1" ] || return 0
        # Tolerate paths containing apostrophes (install.sh emits '\'' for them).
        _exe=$(sed -n "s/^UNSLOTH_EXE='\(.*\)'\$/\1/p" "$1" | head -n1)
        _exe=$(printf '%s' "$_exe" | sed "s/'\\\\''/'/g")
        [ -n "$_exe" ] || return 0
        _emit "$(dirname "$(dirname "$(dirname "$_exe")")")"
    }
    # Mirror install.sh's precedence: UNSLOTH_STUDIO_HOME wins, STUDIO_HOME is ignored when both
    # are set, or uninstalling install A could also delete install B from a leftover STUDIO_HOME.
    # Trimmed, as storage_roots.studio_root() trims them: a whitespace-only override is unset to
    # every resolver, but a bare -n test called it present and suppressed the master-root branch
    # below. _emit then discarded the whitespace path, so an uninstall carrying that environment
    # removed the master root's runtime siblings and left <UNSLOTH_HOME>/studio installed.
    _ush=$(printf '%s' "${UNSLOTH_STUDIO_HOME:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    _sh=$(printf '%s' "${STUDIO_HOME:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    if [ -n "$_ush" ]; then
        _emit "$_ush"
        _from_conf "$_ush/share/studio.conf"
    elif [ -n "$_sh" ]; then
        _emit "$_sh"
        _from_conf "$_sh/share/studio.conf"
    elif [ -n "$(_master_root)" ]; then
        # Last, as in storage_roots.studio_root(): UNSLOTH_HOME names the tree, and the two
        # above name this exact directory, so either of them wins outright.
        _emit "$(_master_root)/studio"
        _from_conf "$(_master_root)/studio/share/studio.conf"
    fi
    # Default-mode conf.
    _from_conf "$HOME/.local/share/unsloth/studio.conf"
}

# Remove $HOME/.local/bin/unsloth only if it is the symlink install.sh created into the studio
# venv. A pip-installed `unsloth` CLI is a regular file: leave it alone rather than wiping an
# unrelated install.
_remove_cli_shim() {
    _shim="$HOME/.local/bin/unsloth"
    [ -L "$_shim" ] || return 0
    _target=$(readlink "$_shim" 2>/dev/null || true)
    case "$_target" in
        */unsloth_studio/bin/unsloth) _remove_path "$_shim" ;;
        *) ;;
    esac
}

# Print key $2 from the Info.plist $1. PlistBuddy also reads binary plists;
# the awk fallback covers XML on hosts without it.
_plist_string() {
    [ -f "$1" ] || return 1
    if [ -x /usr/libexec/PlistBuddy ]; then
        /usr/libexec/PlistBuddy -c "Print :$2" "$1" 2>/dev/null && return 0
    fi
    awk -v k="<key>$2</key>" 'index($0, k) { f = 1; next }
         f && /<string>/ { sub(/.*<string>/, ""); sub(/<\/string>.*/, ""); print; exit }' "$1"
}

# True when the bundle $1 is the packaged desktop app carrying bundle id $2.
# install.sh's shell launcher shares that id, so exclude it by its executable.
_owns_bundle_id() {
    [ -d "$1" ] || return 1
    [ "$(_plist_string "$1/Contents/Info.plist" CFBundleIdentifier)" = "$2" ] || return 1
    [ "$(_plist_string "$1/Contents/Info.plist" CFBundleExecutable)" != "launch-studio" ]
}

# Path of the installed app owning bundle id $1, empty if there is none. A renamed bundle or a
# nested one ("/Applications/AI & ML/Unsloth.app") is supported, so match on the identifier.
_bundle_id_owner() {
    # Overridable so tests can point the scan at a fixture dir.
    _bio_apps="${UNSLOTH_APPLICATIONS_DIR:-/Applications}"
    # Spotlight finds it anywhere on disk. Skipped for a fixture scan: a real
    # install on the machine running the tests must not change the result.
    if [ -z "${UNSLOTH_APPLICATIONS_DIR:-}" ] && command -v mdfind >/dev/null 2>&1; then
        _bio_hit=$(mdfind "kMDItemCFBundleIdentifier == '$1'" 2>/dev/null |
                   while IFS= read -r _bio_app; do
                       _owns_bundle_id "$_bio_app" "$1" && { printf '%s\n' "$_bio_app"; break; }
                   done | head -n 1)
        if [ -n "$_bio_hit" ]; then
            printf '%s\n' "$_bio_hit"
            return 0
        fi
    fi
    # Spotlight can be off or still indexing, so walk the usual roots too. No depth cap:
    # -prune stops the walk at each bundle, so nesting is free and bundles are never entered.
    find "$_bio_apps" "$HOME/Applications" -name '*.app' -type d -prune -print 2>/dev/null |
    while IFS= read -r _bio_app; do
        if _owns_bundle_id "$_bio_app" "$1"; then
            printf '%s\n' "$_bio_app"
            break
        fi
    done | head -n 1
}

_unsloth_uninstall_main() {
    # Reject unknown arguments before destructive work.
    for _arg in "$@"; do
        case "$_arg" in
            -h|--help) _usage; return 0 ;;
            *)
                echo "uninstall.sh: unrecognized argument: $_arg" >&2
                echo "Nothing was removed. Re-run with no arguments to uninstall, or --help." >&2
                return 2
                ;;
        esac
    done

    _uid=$(id -u 2>/dev/null || echo 0)
    _os=$(uname 2>/dev/null || echo unknown)
    _is_wsl=0
    [ "$_os" = "Linux" ] && grep -qi microsoft /proc/version 2>/dev/null && _is_wsl=1

    echo "Stopping any running Unsloth Studio servers..."
    _pkill_studio

    echo "Removing data and install directories..."
    _custom_studio_roots | while IFS= read -r _custom_root; do
        [ -n "$_custom_root" ] || continue
        if _is_unsafe_root "$_custom_root"; then
            echo "  refusing to remove unsafe path: $_custom_root" >&2
            # install.sh accepts any writable root, so a real install can sit under a deny-listed
            # path (/var/tmp/studio). Nothing is deleted and it holds studio.db, so say so.
            [ -d "$_custom_root" ] && _set_marker "$_REMOVE_FAILED_FLAG"
            continue
        fi
        if ! _is_studio_root "$_custom_root"; then
            # Not ours, so skipping leaves none of the user's data behind.
            echo "  refusing to remove non-Unsloth path: $_custom_root" >&2
            continue
        fi
        _remove_root_recording_db "$_custom_root"
        # Native diffusion now installs UNDER the custom root, so the removal above already took
        # it. Older builds put it BESIDE the root at <parent>/stable-diffusion.cpp, which removing
        # the root alone would leave behind. <parent> is user-chosen and "stable-diffusion.cpp" is
        # exactly what `git clone` of the upstream project produces, so require our owner marker
        # (install_sd_cpp_prebuilt) before rm: an unowned checkout or a pre-marker build is kept,
        # never a user file deleted. The derived parent path gets the deny-list check too.
        _custom_sd_cpp="$(dirname "$_custom_root")/stable-diffusion.cpp"
        if _is_unsafe_root "$_custom_sd_cpp"; then
            echo "  refusing to remove unsafe path: $_custom_sd_cpp" >&2
        elif [ -e "$_custom_sd_cpp" ] && [ ! -f "$_custom_sd_cpp/.unsloth-studio-owned" ]; then
            echo "  keeping sd.cpp without Unsloth owner marker: $_custom_sd_cpp" >&2
        else
            _remove_path "$_custom_sd_cpp"
        fi
    done
    # The lexical parent as well. A home that is itself a symlink has its old sd.cpp tree beside
    # the LINK, which the canonicalized loop above never saw. Marker only, no "keeping" notice: an
    # unmarked directory here is somebody's checkout, and the canonical pass already reported.
    _custom_studio_roots lexical 2>/dev/null | while IFS= read -r _lex_root; do
        [ -n "$_lex_root" ] || continue
        # The same ownership check the canonical loop makes. A stale or mistyped
        # UNSLOTH_STUDIO_HOME still reaches here (the lexical pass has no cd -P to filter it), and
        # without this "/parent/typo" would take somebody else's marked sd.cpp with it.
        _is_studio_root "$_lex_root" || continue
        _lex_sd_cpp="$(dirname "$_lex_root")/stable-diffusion.cpp"
        [ -f "$_lex_sd_cpp/.unsloth-studio-owned" ] || continue
        # The deny list is string-based, so it has to see the RESOLVED path: the lexical form can
        # carry ".." or a symlinked ancestor and slip a protected tree past it. The rm stays lexical.
        # shellcheck disable=SC1007
        _lex_sd_canon=$(CDPATH= cd -P -- "$_lex_sd_cpp" 2>/dev/null && pwd -P)
        [ -n "$_lex_sd_canon" ] || _lex_sd_canon="$_lex_sd_cpp"
        if _is_unsafe_root "$_lex_sd_cpp" || _is_unsafe_root "$_lex_sd_canon"; then
            echo "  refusing to remove unsafe path: $_lex_sd_cpp" >&2
        else
            _remove_path "$_lex_sd_cpp"
        fi
    done
    # The master root's own children. Marker-gated and deny-listed rather than removed outright
    # like the ~/.unsloth ones below: <master> is a directory the user chose and may hold their
    # files, so only a tree an Unsloth installer marked is ours to delete. The locks and .staging
    # are ours by name (prebuilt_core.py) and carry no marker.
    _mr_root="$(_master_root)"
    if [ -n "$_mr_root" ]; then
        if _is_unsafe_root "$_mr_root"; then
            echo "  refusing to remove unsafe path: $_mr_root" >&2
        else
            for _mr_child in llama.cpp node whisper.cpp stable-diffusion.cpp; do
                _mr_path="$_mr_root/$_mr_child"
                if _is_unsafe_root "$_mr_path"; then
                    echo "  refusing to remove unsafe path: $_mr_path" >&2
                # -L as well as -e: -e is false for a dangling symlink, so one named llama.cpp
                # with its target volume unmounted fell through to _remove_path, which treats
                # -L as present and unlinks it. In a root the user chose, that link is theirs.
                elif { [ -e "$_mr_path" ] || [ -L "$_mr_path" ]; } \
                    && [ ! -f "$_mr_path/.unsloth-studio-owned" ]; then
                    echo "  keeping $_mr_child without Unsloth owner marker: $_mr_path" >&2
                else
                    _remove_path "$_mr_path"
                fi
            done
            for _mr_lock in .llama.cpp.install.lock .node.install.lock \
                    .whisper.cpp.install.lock .sd.cpp.install.lock; do
                _remove_lock_file "$_mr_root/$_mr_lock"
            done
            # The prebuilt installers SHARE <root>/.staging and prune it only when empty, so
            # anything left in it here is not ours. rmdir, not _remove_path: in a user-chosen
            # root a recursive delete would take files an install was content to leave.
            rmdir "$_mr_root/.staging" 2>/dev/null || true
            for _mr_stale in "$_mr_root"/.*.install.lock.stale.*; do
                [ -e "$_mr_stale" ] && _remove_lock_file "$_mr_stale"
            done
            # Only when nothing of the user's is left; rmdir refuses a non-empty directory.
            rmdir "$_mr_root" 2>/dev/null || true
        fi
    fi
    # end master-root children
    # Same gate as a custom root: an ungated run takes a hand-made ~/.unsloth/studio, and then
    # ~/.unsloth via the empty-dir prune below.
    # -e OR -L: -e follows a link and misses a dangling one, which _remove_path would still
    # unlink, so the gate has to see every entry that exists at that path.
    if { [ -e "$HOME/.unsloth/studio" ] || [ -L "$HOME/.unsloth/studio" ]; } \
       && ! _is_studio_root "$HOME/.unsloth/studio" managed; then
        echo "  refusing to remove non-Unsloth path: $HOME/.unsloth/studio" >&2
        # A refused CUSTOM root is somebody else's by definition. This is our own default path,
        # where a damaged install can sit, so a studio.db here is chat history.
        if [ -f "$HOME/.unsloth/studio/studio.db" ]; then
            _set_marker "$_DB_KEPT_FLAG"
        fi
    else
        _remove_root_recording_db "$HOME/.unsloth/studio"
    fi
    # Shared llama.cpp build + cache, siblings of studio in default mode (deleting studio misses
    # them). No-op in env/custom mode and when absent. A user-set UNSLOTH_LLAMA_CPP_PATH is kept.
    _remove_path "$HOME/.unsloth/llama.cpp"
    # Default-mode native diffusion build, a sibling of studio like llama.cpp. No-op in env/custom
    # mode and when absent. "stable-diffusion.cpp" is exactly what a `git clone` of
    # leejet/stable-diffusion.cpp produces and a user may keep their own checkout (or point
    # UNSLOTH_SD_CPP_PATH) here, so require our owner marker (install_sd_cpp_prebuilt) before rm,
    # mirroring the custom-root guard above.
    _default_sd_cpp="$HOME/.unsloth/stable-diffusion.cpp"
    if [ -e "$_default_sd_cpp" ] && [ ! -f "$_default_sd_cpp/.unsloth-studio-owned" ]; then
        echo "  keeping sd.cpp without Unsloth owner marker: $_default_sd_cpp" >&2
    else
        _remove_path "$_default_sd_cpp"
    fi
    _remove_path "$HOME/.unsloth/.cache"
    # Isolated Node.js runtime (install_node_prebuilt.py), a default-mode sibling of studio.
    _remove_path "$HOME/.unsloth/node"
    # llama.cpp atomic-install staging root (install_llama_prebuilt.py). Normally pruned after
    # activate, but an interrupted build leaves it behind and it blocks the rmdir below.
    _remove_path "$HOME/.unsloth/.staging"
    # Managed whisper.cpp dictation engine (install_whisper_prebuilt.py), a default-mode sibling.
    # Only present when a prebuilt matching the pinned llama.cpp build existed at install time.
    _remove_path "$HOME/.unsloth/whisper.cpp"
    # Prebuilt install locks: every prebuilt serializes on <parent>/.<name>.install.lock
    # (prebuilt_core.py), and a stray lock keeps ~/.unsloth from being pruned below.
    _remove_lock_file "$HOME/.unsloth/.llama.cpp.install.lock"
    _remove_lock_file "$HOME/.unsloth/.node.install.lock"
    _remove_lock_file "$HOME/.unsloth/.whisper.cpp.install.lock"
    # Taking over an abandoned lock renames it to .stale.<pid> before unlinking
    # (install_node_prebuilt.py); a crash between the two strands the rename, and a stranded one
    # blocks the rmdir below. Unmatched globs stay literal, hence the existence test.
    for _stale in "$HOME"/.unsloth/.*.install.lock.stale.*; do
        [ -e "$_stale" ] && _remove_lock_file "$_stale"
    done
    # ROCm-on-WSL helper artifacts (librocdxg clone, smoke-test venv); removing them frees the rmdir.
    _remove_path "$HOME/.unsloth/librocdxg"
    _remove_path "$HOME/.unsloth/rocm-smoketest"
    # Drop ~/.unsloth only if now empty (rmdir refuses non-empty, so user content is kept).
    rmdir "$HOME/.unsloth" 2>/dev/null || true
    _remove_path "$HOME/.local/share/unsloth"
    # CLI shim: only the symlink Unsloth created, never a pip-installed file.
    _remove_cli_shim

    echo "Removing desktop shortcut and launcher lock..."
    # install.sh creates Desktop/Unsloth Studio as a symlink; an unrelated regular dir is kept.
    _desktop_link="$HOME/Desktop/Unsloth Studio"
    if [ -L "$_desktop_link" ] || [ ! -e "$_desktop_link" ]; then
        _remove_path "$_desktop_link"
    else
        echo "  refusing to remove non-symlink Desktop path: $_desktop_link" >&2
    fi
    _remove_path "$HOME/Desktop/unsloth-studio.desktop"
    # Locks are namespaced per-uid; env-mode adds an extra suffix.
    _lock_glob="${XDG_RUNTIME_DIR:-/tmp}/unsloth-studio-launcher-${_uid}"
    for _lock in "$_lock_glob".lock "$_lock_glob"-*.lock; do
        [ -e "$_lock" ] && _remove_path "$_lock"
    done

    case "$_os" in
        Darwin)
            echo "Removing macOS .app bundle and Launch Services entry..."
            _remove_path "$HOME/Applications/Unsloth Studio.app"
            _lsr="/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister"
            if [ -x "$_lsr" ]; then
                "$_lsr" -u "$HOME/Applications/Unsloth Studio.app" 2>/dev/null || true
            fi
            # WKWebView data, keyed by bundle id; created at first launch, not by install.sh. The
            # packaged desktop app shares this id and is the only thing that writes the data, so
            # while that app is still installed this script must not reset it either.
            _bid="ai.unsloth.studio"
            _bid_owner=$(_bundle_id_owner "$_bid")
            if [ -n "$_bid_owner" ]; then
                echo "Keeping app data ($_bid): it belongs to $_bid_owner"
            else
                echo "Removing WebView caches and app data ($_bid)..."
                _remove_path "$HOME/Library/Caches/$_bid"
                _remove_path "$HOME/Library/WebKit/$_bid"
                _remove_path "$HOME/Library/Application Support/$_bid"
                _remove_path "$HOME/Library/HTTPStorages/$_bid"
                _remove_path "$HOME/Library/HTTPStorages/$_bid.binarycookies"
                _remove_path "$HOME/Library/Cookies/$_bid.binarycookies"
                _remove_path "$HOME/Library/Saved Application State/$_bid.savedState"
                # defaults, not rm: cfprefsd rewrites the plist from memory. ByHost is a separate
                # domain. As the home's owner, or under sudo root just edits root's own domain.
                if command -v defaults >/dev/null 2>&1; then
                    _run_as_home_owner defaults delete "$_bid" >/dev/null 2>&1 || true
                    _run_as_home_owner defaults -currentHost delete "$_bid" >/dev/null 2>&1 || true
                fi
                _remove_path "$HOME/Library/Preferences/$_bid.plist"
            fi
            ;;
        Linux)
            if [ "$_is_wsl" = "1" ]; then
                echo "Removing WSL Windows-side shortcuts..."
                # install.sh creates per-distro 'Unsloth Studio (WSL - <distro>).lnk' on the
                # Windows Desktop + Start Menu. Scope removal to THIS distro so a multi-distro
                # install keeps the other distros' launchers; the TARGET=wsl.exe check spares a
                # native install's "Unsloth Studio.lnk". Test powershell.exe can EXECUTE:
                # `command -v` succeeds even with interop OFF and the .exe then fails to run.
                _wsl_distro="${WSL_DISTRO_NAME:-}"
                _ps_ran=0
                if command -v powershell.exe >/dev/null 2>&1 && \
                   powershell.exe -NoProfile -Command "exit 0" >/dev/null 2>&1; then
                    _ps_ran=1
                    # Inject the distro into the command: a -Command string does not receive
                    # trailing tokens as $args. WSL distro names are safe to embed.
                    # shellcheck disable=SC2016
                    powershell.exe -NoProfile -Command '$distro = "'"$_wsl_distro"'";
                        $dirs = @(
                            [Environment]::GetFolderPath("Desktop"),
                            (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs")
                        );
                        $ws = New-Object -ComObject WScript.Shell;
                        foreach ($d in $dirs) {
                            if (-not $d -or -not (Test-Path -LiteralPath $d)) { continue }
                            Get-ChildItem -LiteralPath $d -Filter "Unsloth Studio*.lnk" -ErrorAction SilentlyContinue | ForEach-Object {
                                try {
                                    $sc = $ws.CreateShortcut($_.FullName);
                                    if ("$($sc.TargetPath) $($sc.Arguments)" -notmatch "wsl\.exe") { return }
                                    # When the distro is known, require the per-distro
                                    # name for this distro or its -d "<distro>" argument
                                    # so launchers for other distros are not removed.
                                    if ($distro) {
                                        $nameMatch = ($_.Name -eq "Unsloth Studio (WSL - $distro).lnk");
                                        $argMatch  = ($sc.Arguments -match ("-d\s+`"?" + [regex]::Escape($distro) + "`"?"));
                                        if (-not ($nameMatch -or $argMatch)) { return }
                                    }
                                    Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
                                } catch { }
                            }
                        }
                        # Keep the shared icon while any Unsloth shortcut still uses it (native
                        # install or another WSL distro); drop it only with the last one.
                        $iconInUse = $false;
                        foreach ($d in $dirs) {
                            if (-not $d -or -not (Test-Path -LiteralPath $d)) { continue }
                            if (Get-ChildItem -LiteralPath $d -Filter "Unsloth Studio*.lnk" -ErrorAction SilentlyContinue) { $iconInUse = $true; break }
                        }
                        # Guard LOCALAPPDATA: empty on a service/SYSTEM account makes
                        # Join-Path throw, aborting the icon cleanup (mirror uninstall.ps1).
                        if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
                            $iconDir = Join-Path $env:LOCALAPPDATA "Unsloth Studio";
                            $ico = Join-Path $iconDir "unsloth.ico";
                            if ((-not $iconInUse) -and (Test-Path -LiteralPath $ico)) { Remove-Item -LiteralPath $ico -Force -ErrorAction SilentlyContinue }
                            if ((Test-Path -LiteralPath $iconDir) -and -not (Get-ChildItem -LiteralPath $iconDir -Force -ErrorAction SilentlyContinue)) { Remove-Item -LiteralPath $iconDir -Recurse -Force -ErrorAction SilentlyContinue }
                        }' >/dev/null 2>&1 || true
                fi
                # Remove $1's shared unsloth.ico only if no Unsloth shortcut (native install or
                # another WSL distro) still uses it, then drop the dir if empty. Reciprocal of
                # uninstall.ps1's _RemoveDataDirKeepingWslIcon.
                _drop_shared_icon_if_unused() {
                    _du="$1"
                    _icodir="$_du/AppData/Local/Unsloth Studio"
                    _icon_in_use=0
                    for _sd in \
                        "$_du/Desktop" \
                        "$_du/OneDrive/Desktop" \
                        "$_du"/OneDrive*/Desktop \
                        "$_du/AppData/Roaming/Microsoft/Windows/Start Menu/Programs"; do
                        [ -d "$_sd" ] || continue
                        for _any in "$_sd"/"Unsloth Studio"*.lnk; do
                            [ -e "$_any" ] && { _icon_in_use=1; break; }
                        done
                        [ "$_icon_in_use" = "1" ] && break
                    done
                    if [ "$_icon_in_use" = "0" ]; then
                        [ -f "$_icodir/unsloth.ico" ] && rm -f "$_icodir/unsloth.ico" 2>/dev/null || true
                    fi
                    [ -d "$_icodir" ] && rmdir "$_icodir" 2>/dev/null || true
                }
                # Fallback when interop is off: remove WSL .lnk files via drvfs. The
                # "Unsloth Studio (WSL..." name never matches a native "Unsloth Studio.lnk".
                if [ "$_ps_ran" = "0" ]; then
                    for _drive in /mnt/c /mnt/d /mnt/e; do
                        [ -d "$_drive/Users" ] || continue
                        for _udir in "$_drive"/Users/*; do
                            [ -d "$_udir" ] || continue
                            for _scdir in \
                                "$_udir/Desktop" \
                                "$_udir/OneDrive/Desktop" \
                                "$_udir"/OneDrive*/Desktop \
                                "$_udir/AppData/Roaming/Microsoft/Windows/Start Menu/Programs"; do
                                [ -d "$_scdir" ] || continue
                                if [ -n "$_wsl_distro" ]; then
                                    # Exact per-distro name (no glob) so other distros survive.
                                    _lnk="$_scdir/Unsloth Studio (WSL - ${_wsl_distro}).lnk"
                                    [ -e "$_lnk" ] && rm -f "$_lnk" 2>/dev/null && echo "  removed: $_lnk" || true
                                else
                                    # Distro unknown: fall back to the broad WSL prefix.
                                    for _lnk in "$_scdir"/"Unsloth Studio (WSL"*.lnk; do
                                        [ -e "$_lnk" ] && rm -f "$_lnk" 2>/dev/null && echo "  removed: $_lnk" || true
                                    done
                                fi
                            done
                            # Drop the shared icon only when no shortcut still needs it.
                            _drop_shared_icon_if_unused "$_udir"
                        done
                    done
                fi
                # ── ROCm-on-WSL config (install_rocm_wsl_strixhalo.sh) ──
                # Remove Unsloth's own ROCDXG config (the env it persisted). The system
                # ROCm userspace is a shared prereq (like CUDA) and is LEFT IN PLACE by
                # default; set UNSLOTH_UNINSTALL_ROCM=1 to remove it too.
                echo "Removing ROCm-on-WSL config..."
                _sudo=""
                if [ "$_uid" != "0" ] && command -v sudo >/dev/null 2>&1; then _sudo="sudo"; fi
                $_sudo rm -f /etc/profile.d/unsloth-rocm-wsl.sh 2>/dev/null || true
                if [ -f "$HOME/.bashrc" ] && grep -q "Unsloth ROCm-on-WSL" "$HOME/.bashrc" 2>/dev/null; then
                    _bk=$(mktemp 2>/dev/null || echo "$HOME/.bashrc.unsloth.tmp")
                    if sed '/# >>> Unsloth ROCm-on-WSL/,/# <<< Unsloth ROCm-on-WSL/d' "$HOME/.bashrc" > "$_bk" 2>/dev/null; then
                        cat "$_bk" > "$HOME/.bashrc" 2>/dev/null || true
                        echo "  cleaned ROCm-on-WSL block from ~/.bashrc"
                    fi
                    rm -f "$_bk" 2>/dev/null || true
                fi
                if [ "${UNSLOTH_UNINSTALL_ROCM:-0}" = "1" ]; then
                    echo "  removing system ROCm (UNSLOTH_UNINSTALL_ROCM=1)..."
                    $_sudo rm -f /etc/apt/sources.list.d/rocm.list /etc/apt/preferences.d/rocm-pin-600 \
                        /etc/apt/keyrings/rocm.gpg /etc/ld.so.conf.d/rocm.conf 2>/dev/null || true
                    $_sudo sh -c 'rm -rf /opt/rocm /opt/rocm-*' 2>/dev/null || true
                    if command -v ldconfig >/dev/null 2>&1; then $_sudo ldconfig 2>/dev/null || true; fi
                elif [ -d /opt/rocm ]; then
                    echo "  Note: ROCm userspace (/opt/rocm*) left in place (shared prereq)."
                    echo "        Remove it by re-running with UNSLOTH_UNINSTALL_ROCM=1, or manually:"
                    echo "          sudo rm -rf /opt/rocm /opt/rocm-* && sudo ldconfig"
                fi
            fi
            # webkit2gtk data by bundle id: Tauri points the WebView at LocalData/<bid>, so the
            # caches sit under XDG_DATA_HOME; the rest is app data.
            _bid="ai.unsloth.studio"
            echo "Removing WebView caches and app data ($_bid)..."
            _remove_path "$(_xdg_dir "${XDG_DATA_HOME:-}" "$HOME/.local/share")/$_bid"
            _remove_path "$(_xdg_dir "${XDG_CACHE_HOME:-}" "$HOME/.cache")/$_bid"
            _remove_path "$(_xdg_dir "${XDG_CONFIG_HOME:-}" "$HOME/.config")/$_bid"
            _remove_path "$(_xdg_dir "${XDG_STATE_HOME:-}" "$HOME/.local/state")/$_bid"
            echo "Removing Linux .desktop entry..."
            _remove_path "$HOME/.local/share/applications/unsloth-studio.desktop"
            # tauri-plugin-deep-link rewrites "<exe>-handler.desktop" on every launch for the
            # unsloth:// scheme, so it exists on any machine the app has started on. Unlike
            # install.sh's own shortcut it uses Tauri's data_dir(), which honours XDG_DATA_HOME.
            _un_appdir="$(_xdg_dir "${XDG_DATA_HOME:-}" "$HOME/.local/share")/applications"
            _remove_path "$_un_appdir/unsloth-studio-handler.desktop"
            if [ "$_un_appdir" != "$HOME/.local/share/applications" ]; then
                _remove_path "$HOME/.local/share/applications/unsloth-studio-handler.desktop"
            fi
            # Rebuild mimeinfo.cache wherever an entry was removed, or it keeps advertising it.
            if command -v update-desktop-database >/dev/null 2>&1; then
                update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
                if [ "$_un_appdir" != "$HOME/.local/share/applications" ]; then
                    update-desktop-database "$_un_appdir" 2>/dev/null || true
                fi
            fi
            ;;
    esac

    echo ""
    echo "Unsloth Studio uninstalled."
    if _markers_unavailable || _marker_set "$_REMOVE_FAILED_FLAG"; then
        # Also the no-marker-storage case: no record of a failed rm, so do not claim success.
        echo "Note: some paths could not be removed (see 'could not remove:' above), so the"
        echo "      signed-in session and local chat history may still be on disk. Remove"
        echo "      those paths by hand to clear them."
    elif _marker_set "$_DB_REMOVED_FLAG"; then
        # Scoped to what was removed: a default and an env-mode install can coexist, and a bare
        # run never discovers the custom root, so "are gone" would be false for its database.
        echo "Note: this also removed the app's WebView data and the studio.db it found, so"
        echo "      the desktop app's session and the chat history in the install(s) removed"
        echo "      above are gone."
    else
        # No studio.db was deleted, so only the WebView-local data is accounted for: an env-mode
        # install this run never discovered still has its keys and history.
        echo "Note: this also removed the app's WebView data, so the desktop app's session is"
        echo "      gone. A browser session is not affected: its tokens live in the same"
        echo "      localStorage as the API keys below."
        if _marker_set "$_DB_KEPT_FLAG"; then
            # Named, and with no advice to delete it: the gate kept it precisely because it
            # does not look like ours.
            echo "      $HOME/.unsloth/studio carries no Unsloth install marker, so it was left"
            echo "      alone. The studio.db inside it is still there; look at that directory"
            echo "      yourself before deciding what to do with it."
        else
            echo "      No studio.db was found, so any chat history in an install root this run"
            echo "      did not see is still on disk."
        fi
    fi
    echo "Note: provider API keys are kept in the browser's localStorage, not in studio.db."
    echo "      Unless you ran Unsloth as the desktop app, clear site data for the"
    echo "      http://localhost:<port> origin you used to remove them."
    echo "Note: Hugging Face model cache at ~/.cache/huggingface was left in place."
    echo "Remove it manually with 'rm -rf ~/.cache/huggingface/hub' if desired."
    # Env-mode installs leave no breadcrumb in $HOME, so a custom root is only found when the
    # user re-exports the variable. Hint when neither is set, so `curl | sh` does not silently miss.
    if [ -z "${UNSLOTH_STUDIO_HOME:-}" ] && [ -z "${STUDIO_HOME:-}" ]; then
        echo ""
        echo "If you installed Unsloth Studio with UNSLOTH_STUDIO_HOME or STUDIO_HOME"
        echo "pointing at a custom directory, re-run this script with the same variable"
        echo "set to also remove that install tree, e.g.:"
        echo "  UNSLOTH_STUDIO_HOME=/your/path sh -c \"\$(curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh)\""
    fi
}

# Parse the entire script before running destructive work. This keeps piped help
# from closing the writer early and makes a truncated download inert.
{
    _unsloth_uninstall_main "$@"
}
