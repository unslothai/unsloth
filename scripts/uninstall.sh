#!/usr/bin/env sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unsloth Studio uninstaller (macOS / Linux / WSL). Run --help for details.
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
beside the install dir: ~/.unsloth/{llama.cpp,node,whisper.cpp,audio.cpp,.cache}.
The Hugging Face cache at ~/.cache/huggingface is left in place (only audio.cpp's
unsloth-audiocpp-links beside it goes), as is anything else you keep under
~/.unsloth. A shared uv package cache (`uv cache dir`) is also left when install
reused one.

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

_kill_pid_file() {
    _pid_file="$1"
    [ -f "$_pid_file" ] || return 0
    _pid=$(sed -n '1s/[^0-9].*//p' "$_pid_file" 2>/dev/null || true)
    if [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null; then
        kill -TERM "$_pid" 2>/dev/null || true
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

# The nested <root>/stable-diffusion.cpp is not marker-gated since this run deletes that root;
# the default and legacy sibling paths survive when unowned, so they are.
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

# Lexical roots differ from canonical ones only when the Unsloth home is a symlink;
# old builds used plain dirname.
_sd_cpp_sibling_bases() {
    {
        _custom_studio_roots 2>/dev/null
        _custom_studio_roots lexical 2>/dev/null
    } | awk '!seen[$0]++'
}

# A live sd-server keeps running after its binary is unlinked, so stop it first.
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

# Owner of $HOME, not the caller: macOS sudo keeps HOME, so euid can be 0 here.
_home_uid() {
    # -L: stat lstats by default, so a root-owned link to a user home would read as uid 0.
    _hu=$(stat -L -c %u "$HOME" 2>/dev/null || stat -L -f %u "$HOME" 2>/dev/null || true)
    case "$_hu" in ''|*[!0-9]*) _hu=$(id -u 2>/dev/null || true) ;; esac
    case "$_hu" in *[!0-9]*) _hu= ;; esac
    printf '%s\n' "$_hu"
}

# Run as the $HOME owner when root, so per-user daemons see the right domain.
_run_as_home_owner() {
    _ro=$(_home_uid)
    if [ "$(id -u 2>/dev/null || echo 0)" = "0" ] && [ -n "$_ro" ] && [ "$_ro" != "0" ] &&
       command -v launchctl >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1; then
        launchctl asuser "$_ro" sudo -u "#$_ro" "$@"
    else
        "$@"
    fi
}

# Only reached without pkill. Scoped to the $HOME owner, matching pkill -u below.
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
    for _data_dir in "$HOME/.local/share/unsloth" $(_custom_studio_data_dirs); do
        [ -d "$_data_dir" ] || continue
        for _pf in "$_data_dir"/studio-*.pid; do
            [ -f "$_pf" ] && _kill_pid_file "$_pf"
        done
    done

    if ! command -v pkill >/dev/null 2>&1; then
        # No procps: a live app re-creates the profile after the delete, so do not claim a clean removal.
        if _studio_app_running; then
            echo "  pkill not found and Unsloth Studio is running; close it and re-run" >&2
            _set_marker "$_REMOVE_FAILED_FLAG"
        fi
        return 0
    fi

    # Scope patterns to the roots being removed so a different install is not touched.
    _kill_roots="$HOME/.unsloth/studio"
    _roots_from_conf=$(_custom_studio_roots 2>/dev/null || true)
    [ -n "$_roots_from_conf" ] && _kill_roots="$_kill_roots
$_roots_from_conf"

    printf '%s\n' "$_kill_roots" | while IFS= read -r _root; do
        [ -n "$_root" ] || continue
        [ -d "$_root" ] || continue
        _re=$(_pkill_escape "$_root")
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

    _stop_owned_sd_cpp_processes TERM
    sleep 0.5
    _stop_owned_sd_cpp_processes KILL

    # WebView helpers re-create caches, so kill the app. -x spares the CLI shim; -u is the $HOME owner
    # (macOS sudo keeps HOME). Numeric uid and signal first for BSD pkill.
    _studio_uid=$(_home_uid)
    if [ -n "$_studio_uid" ]; then
        pkill -TERM -x -u "$_studio_uid" unsloth-studio 2>/dev/null || true
        sleep 0.5
        pkill -KILL -x -u "$_studio_uid" unsloth-studio 2>/dev/null || true
    fi
}

# Markers in files, not variables: custom roots are removed in a pipeline subshell.
# db-kept is separate from remove-failed: the refused default root failed nothing.
# mktemp -d: private (0700) and unpredictable, so nothing else can pre-create the markers.
_MARKER_DIR=$(mktemp -d 2>/dev/null || true)
_REMOVE_FAILED_FLAG=""
_DB_REMOVED_FLAG=""
_DB_KEPT_FLAG=""
_UV_ROOTS_FILE=""
_UV_LEFTOVER_FILE=""
_UV_SAW_MARKER_FLAG=""
if [ -n "$_MARKER_DIR" ] && [ -d "$_MARKER_DIR" ]; then
    _REMOVE_FAILED_FLAG="$_MARKER_DIR/remove-failed"
    _DB_REMOVED_FLAG="$_MARKER_DIR/db-removed"
    _DB_KEPT_FLAG="$_MARKER_DIR/db-kept"
    _UV_ROOTS_FILE="$_MARKER_DIR/uv-roots"
    _UV_LEFTOVER_FILE="$_MARKER_DIR/uv-leftovers"
    _UV_SAW_MARKER_FLAG="$_MARKER_DIR/uv-saw-marker"
fi

# printf, not `: >`: a redirect error on a special builtin kills dash/ash despite || true.
_set_marker() {
    [ -n "$1" ] || return 0
    printf '' > "$1" 2>/dev/null || true
    return 0
}
_marker_set() { [ -n "$1" ] && [ -f "$1" ]; }
# Re-checked each time: the dir can vanish mid-run and _set_marker would silently drop failures.
_markers_unavailable() {
    [ -n "$_MARKER_DIR" ] || return 0
    [ -d "$_MARKER_DIR" ] || return 0
    [ -w "$_MARKER_DIR" ] || return 0
    return 1
}

# Read before any root is deleted; a uv cache outside removed roots is shared and stays.
_uv_cache_under_any_root() {
    while IFS= read -r _uv_r; do
        [ -L "$_uv_r" ] && continue
        case "$1" in "$_uv_r"|"$_uv_r"/*) return 0 ;; esac
    done < "$_UV_ROOTS_FILE"
    return 1
}

_uv_collect_from_install_roots() {
    [ -n "$_UV_ROOTS_FILE" ] || return 0
    {
        printf '%s\n' "$HOME/.unsloth/studio"
        _custom_studio_roots | while IFS= read -r _uv_root; do
            [ -n "$_uv_root" ] || continue
            if ! _is_unsafe_root "$_uv_root" && _is_studio_root "$_uv_root"; then
                printf '%s\n' "$_uv_root"
            fi
        done
    } > "$_UV_ROOTS_FILE" 2>/dev/null || return 0
    while IFS= read -r _uv_root; do
        [ -f "$_uv_root/cache/uv-cache-dir" ] || continue
        _set_marker "$_UV_SAW_MARKER_FLAG"
        _uv_rec=$(sed -n '1p' "$_uv_root/cache/uv-cache-dir" 2>/dev/null | tr -d '\r') || _uv_rec=""
        [ -n "$_uv_rec" ] || continue
        _uv_cache_under_any_root "$_uv_rec" || printf '%s\n' "$_uv_rec" >> "$_UV_LEFTOVER_FILE" 2>/dev/null || true
    done < "$_UV_ROOTS_FILE"
}

_uv_print_leftover_notes() {
    if [ -s "$_UV_LEFTOVER_FILE" ]; then
        awk '!seen[$0]++' "$_UV_LEFTOVER_FILE" 2>/dev/null | while IFS= read -r _uv_path; do
            [ -n "$_uv_path" ] || continue
            # Explicit --cache-dir: bare `uv cache clean` cleans whatever cache uv resolves now.
            _uv_q=$(printf '%s' "$_uv_path" | sed "s/'/'\\\\''/g")
            echo "Note: the uv package cache at $_uv_path was left in place (it may be shared with other tools)."
            echo "      Free it with: uv cache clean --cache-dir '$_uv_q'"
        done
    elif ! _marker_set "$_UV_SAW_MARKER_FLAG"; then
        echo "Note: if install reused a shared uv cache (\`uv cache dir\`), it was left in place."
        echo "      Free it with 'uv cache clean'."
    fi
}

# EXIT trap: --help, a bad argument and set -e all skip an end-of-main cleanup.
_cleanup_markers() {
    if [ -n "$_MARKER_DIR" ]; then
        rm -rf "$_MARKER_DIR" 2>/dev/null || true
    fi
}
trap _cleanup_markers EXIT

# A partial removal can drop the sentinels; restore the marker so a retry's gate accepts the root.
_restore_owner_marker() {
    [ -d "$1" ] || return 0
    if [ -e "$1/.unsloth-studio-owned" ] || [ -L "$1/.unsloth-studio-owned" ]; then return 0; fi
    printf '' > "$1/.unsloth-studio-owned" 2>/dev/null || true
    return 0
}

# The db check uses the resolved path: rm -rf only unlinks a symlinked root.
# Never follow the link to delete its target; that is what the deny lists prevent.
_remove_root_recording_db() {
    _rrd_root="$1"
    # shellcheck disable=SC1007
    _rrd_real=$(CDPATH= cd -P -- "$_rrd_root" 2>/dev/null && pwd -P) || _rrd_real=""
    [ -n "$_rrd_real" ] || _rrd_real="$_rrd_root"
    _rrd_had_db=0
    _rrd_db="$_rrd_real/studio.db"
    if [ -f "$_rrd_db" ]; then
        _rrd_had_db=1
        # A symlinked db survives rm of the link. No readlink -f (BSD lacks it before macOS 12.3).
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
            _set_marker "$_REMOVE_FAILED_FLAG"
        fi
    fi
}

# Locks are always regular files (O_CREAT|O_EXCL); a dir or link here is the user's.
# Must agree with uninstall.ps1's _RemoveLockFile.
_remove_lock_file() {
    _rlf="$1"
    if [ -L "$_rlf" ]; then
        echo "  keeping link at an install-lock path: $_rlf" >&2
    elif [ -f "$_rlf" ]; then
        _remove_path "$_rlf"
    elif [ -e "$_rlf" ]; then
        echo "  keeping non-file at an install-lock path: $_rlf" >&2
    fi
}

# A relative XDG override is invalid and ignored by Tauri, so honouring it would rm under cwd.
_xdg_dir() {
    case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$2" ;; esac
}

# Root accepted only with an install-time owner marker (matches install.sh's env-mode guard).
# Plain -f, following links: a relocated venv keeps a real marker behind a link.
_is_owner_marker() { [ -f "$1" ]; }

# No -L: a relocated venv is still a venv.
_is_venv_dir() {
    [ -d "$1" ] || return 1
    [ -f "$1/pyvenv.cfg" ] && return 0
    [ -f "$1/bin/python" ] && return 0
    return 1
}

# Exact installer name <prefix>.<stamp>.<pid>[.<n>]; must not be looser than install.sh.
_is_installer_leftover_name() {
    _l=${1##*/}
    # Only the rollback name carries a collision counter; .venv.invalid takes no suffix.
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
    _managed="${2:-}"
    [ -n "$_r" ] || return 1
    # install.sh writes the marker before the venv, so partial installs identify themselves.
    _is_owner_marker "$_r/.unsloth-studio-owned" && return 0
    _is_owner_marker "$_r/share/studio.conf" && return 0
    _is_owner_marker "$_r/unsloth_studio/.unsloth-studio-owned" && return 0
    _is_owner_marker "$_r/.venv/.unsloth-studio-owned" && return 0
    if [ -L "$_r/bin/unsloth" ]; then
        _t=$(readlink "$_r/bin/unsloth" 2>/dev/null || true)
        case "$_t" in *unsloth_studio/bin/unsloth) return 0 ;; esac
    fi
    # bin/unsloth alone is weak proof (any venv with the wheel has it), so only trust it at the managed root.
    [ "$_managed" = managed ] || return 1
    for _v in unsloth_studio .venv; do
        _is_venv_dir "$_r/$_v" || continue
        [ -f "$_r/$_v/bin/unsloth" ] && return 0
    done
    # An install that died between moving the venv aside and writing the marker leaves only these.
    for _p in "$_r"/unsloth_studio.rollback.* "$_r"/.venv.invalid.*; do
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

_custom_studio_data_dirs() {
    _custom_studio_roots 2>/dev/null | while IFS= read -r _r; do
        [ -d "$_r/share" ] && printf '%s\n' "$_r/share"
    done
}

# The master root UNSLOTH_HOME names, or empty. Trimmed and tilde-expanded like
# storage_roots.unsloth_home().
_master_root() {
    _mr_from_note=
    _mr=$(printf '%s' "${UNSLOTH_HOME:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    # Fall back to the note setup.sh leaves in the Studio tree, only for the root this run targets,
    # in studio_root() precedence, or two installs could cross-delete runtimes.
    if [ -z "$_mr" ]; then
        # Trimmed as setup.sh trims it, or a padded value never finds the note.
        _mr_ush=$(printf '%s' "${UNSLOTH_STUDIO_HOME:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        _mr_sh=$(printf '%s' "${STUDIO_HOME:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        if [ -n "$_mr_ush" ]; then
            _mr_roots="$_mr_ush"
        elif [ -n "$_mr_sh" ]; then
            _mr_roots="$_mr_sh"
        else
            # Not via _custom_studio_roots, which calls back into this function. Root is three dirnames
            # up from UNSLOTH_EXE; apostrophes arrive escaped as '\''.
            _mr_roots=""
            _mr_dconf="$HOME/.local/share/unsloth/studio.conf"
            if [ -f "$_mr_dconf" ]; then
                _mr_exe=$(sed -n "s/^UNSLOTH_EXE='\(.*\)'\$/\1/p" "$_mr_dconf" | head -n1)
                _mr_exe=$(printf '%s' "$_mr_exe" | sed "s/'\\\\''/'/g")
                [ -n "$_mr_exe" ] && _mr_roots=$(dirname "$(dirname "$(dirname "$_mr_exe")")")
            fi
            _mr_roots="$_mr_roots
$HOME/.unsloth/studio"
        fi
        _mr_saved_ifs=$IFS
        IFS='
'
        for _mr_studio in $_mr_roots; do
            IFS=$_mr_saved_ifs
            [ -n "$_mr_studio" ] || continue
            _mr_conf="${_mr_studio}/share/.unsloth-master-root"
            [ -f "$_mr_conf" ] || continue
            # A multi-line note is refused: storage_roots.py and the CLI reject it too, so it
            # must not license a delete.
            if [ -n "$(sed -n '2,$p' "$_mr_conf" 2>/dev/null | tr -d '[:space:]')" ]; then
                continue
            fi
            _mr=$(head -n 1 "$_mr_conf" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//') \
                || _mr=""
            if [ -n "$_mr" ]; then
                _mr_from_note=1
                break
            fi
        done
        IFS=$_mr_saved_ifs
    fi
    [ -n "$_mr" ] || return 0
    # shellcheck disable=SC2088
    case "$_mr" in
        "~") _mr="$HOME" ;;
        "~/"*) _mr="$HOME/${_mr#'~/'}" ;;
    esac
    # shellcheck disable=SC1007
    # || fallback: dash/POSIX sh apply set -e to a failed substitution in an assignment.
    _mr_canon=$(CDPATH= cd -P -- "$_mr" 2>/dev/null && pwd -P) || _mr_canon=""
    [ -n "$_mr_canon" ] && _mr="$_mr_canon"
    # A note must describe the tree it was found in, or a copied tree would delete the original's runtimes.
    # An explicit UNSLOTH_HOME skips this check.
    if [ "${_mr_from_note:-}" = 1 ]; then
        _mr_here=$(CDPATH= cd -P -- "$_mr_studio" 2>/dev/null && pwd -P) || _mr_here=""
        # A note inside ~/.unsloth/studio is declined outright, matching
        # storage_roots._is_legacy_studio_tree.
        _mr_legacy_studio=$(CDPATH= cd -P -- "$HOME/.unsloth/studio" 2>/dev/null && pwd -P) \
            || _mr_legacy_studio="$HOME/.unsloth/studio"
        [ "$_mr_here" != "$_mr_legacy_studio" ] || return 0
        case "$_mr_here" in
            "$_mr") : ;;
            "$_mr"/*) : ;;
            *) return 0 ;;
        esac
    fi
    case "$_mr" in "$HOME/.unsloth"|/|"") return 0 ;; esac
    printf '%s\n' "$_mr"
}

_custom_studio_roots() {
    # $1 = "lexical" skips canonicalization; reset on every call.
    _studio_roots_lexical="${1:-}"
    _seen=""
    _emit() {
        _r="$1"
        [ -z "$_r" ] && return 0
        # Tilde expansion, matching install.sh's _resolve_studio_destinations.
        # shellcheck disable=SC2088
        case "$_r" in
            "~") _r="$HOME" ;;
            "~/"*) _r="$HOME/${_r#'~/'}" ;;
        esac
        # Canonicalize so variants hit the _is_unsafe_root deny list; the lexical pass
        # rebuilds old dirname paths.
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
        _exe=$(sed -n "s/^UNSLOTH_EXE='\(.*\)'\$/\1/p" "$1" | head -n1)
        _exe=$(printf '%s' "$_exe" | sed "s/'\\\\''/'/g")
        [ -n "$_exe" ] || return 0
        _emit "$(dirname "$(dirname "$(dirname "$_exe")")")"
    }
    # Mirror install.sh: UNSLOTH_STUDIO_HOME wins over STUDIO_HOME. Trimmed like storage_roots.studio_root().
    _ush=$(printf '%s' "${UNSLOTH_STUDIO_HOME:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    _sh=$(printf '%s' "${STUDIO_HOME:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    if [ -n "$_ush" ]; then
        _emit "$_ush"
        _from_conf "$_ush/share/studio.conf"
    elif [ -n "$_sh" ]; then
        _emit "$_sh"
        _from_conf "$_sh/share/studio.conf"
    elif [ -n "$(_master_root)" ]; then
        # Last, as in storage_roots.studio_root().
        _emit "$(_master_root)/studio"
        _from_conf "$(_master_root)/studio/share/studio.conf"
    fi
    # Default-mode conf.
    _from_conf "$HOME/.local/share/unsloth/studio.conf"
}

# Remove the CLI shim only if it is install.sh's symlink; a pip-installed unsloth is a regular file.
_remove_cli_shim() {
    _shim="$HOME/.local/bin/unsloth"
    [ -L "$_shim" ] || return 0
    _target=$(readlink "$_shim" 2>/dev/null || true)
    case "$_target" in
        */unsloth_studio/bin/unsloth) _remove_path "$_shim" ;;
        *) ;;
    esac
}

# PlistBuddy reads binary plists; awk fallback handles XML only.
_plist_string() {
    [ -f "$1" ] || return 1
    if [ -x /usr/libexec/PlistBuddy ]; then
        /usr/libexec/PlistBuddy -c "Print :$2" "$1" 2>/dev/null && return 0
    fi
    awk -v k="<key>$2</key>" 'index($0, k) { f = 1; next }
         f && /<string>/ { sub(/.*<string>/, ""); sub(/<\/string>.*/, ""); print; exit }' "$1"
}

# install.sh's shell launcher shares the bundle id, so exclude it by executable.
_owns_bundle_id() {
    [ -d "$1" ] || return 1
    [ "$(_plist_string "$1/Contents/Info.plist" CFBundleIdentifier)" = "$2" ] || return 1
    [ "$(_plist_string "$1/Contents/Info.plist" CFBundleExecutable)" != "launch-studio" ]
}

# Match on bundle id: renamed and nested bundles are supported.
_bundle_id_owner() {
    # Overridable so tests can point the scan at a fixture dir.
    _bio_apps="${UNSLOTH_APPLICATIONS_DIR:-/Applications}"
    # Skipped for fixture scans so a real install does not change test results.
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
    # Spotlight can be off or indexing. -prune stops at each bundle, so no depth cap is needed.
    find "$_bio_apps" "$HOME/Applications" -name '*.app' -type d -prune -print 2>/dev/null |
    while IFS= read -r _bio_app; do
        if _owns_bundle_id "$_bio_app" "$1"; then
            printf '%s\n' "$_bio_app"
            break
        fi
    done | head -n 1
}

_unsloth_uninstall_main() {
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

    # Before the kill sweep, or Restart=on-failure brings the server back mid-removal.
    _remove_systemd_user_service() {
        _sd_dir="$(_xdg_dir "${XDG_CONFIG_HOME:-}" "$HOME/.config")/systemd/user"
        _sd_unit="$_sd_dir/unsloth-studio.service"
        if [ ! -f "$_sd_unit" ] && command -v systemctl >/dev/null 2>&1; then
            _sd_frag=$(systemctl --user show -p FragmentPath --value unsloth-studio.service 2>/dev/null || true)
            case "$_sd_frag" in */unsloth-studio.service) _sd_unit="$_sd_frag"; _sd_dir="${_sd_frag%/*}" ;; esac
        fi
        if [ ! -f "$_sd_unit" ] && command -v getent >/dev/null 2>&1; then
            _sd_pw=$(getent passwd "$(id -un 2>/dev/null)" 2>/dev/null | cut -d: -f6)
            case "$_sd_pw" in /*) _sd_dir="$_sd_pw/.config/systemd/user"; _sd_unit="$_sd_dir/unsloth-studio.service" ;; esac
        fi
        [ -f "$_sd_unit" ] || return 0
        [ "$(head -n 1 "$_sd_unit" 2>/dev/null)" = "# unsloth-studio-managed-systemd" ] || return 0
        _sd_stopped=0
        if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
            systemctl --user disable --now unsloth-studio.service 2>/dev/null && _sd_stopped=1
        fi
        _remove_path "$_sd_dir/default.target.wants/unsloth-studio.service"
        _remove_path "$_sd_unit"
        [ "$_sd_stopped" = 1 ] && { systemctl --user daemon-reload 2>/dev/null || true; }
        echo "Removed systemd user service (unsloth-studio.service)."
        [ "$_sd_stopped" = 1 ] || echo "  could not reach the systemd user manager to stop it; it will not start again after the next login or reboot" >&2
    }
    _remove_systemd_user_service

    echo "Stopping any running Unsloth Studio servers..."
    _pkill_studio

    echo "Removing data and install directories..."
    _uv_collect_from_install_roots
    # Resolved once before deletion: the note it may read lives in a tree the loop removes.
    _MASTER_ROOT_SAVED="$(_master_root)"
    _custom_studio_roots | while IFS= read -r _custom_root; do
        [ -n "$_custom_root" ] || continue
        if _is_unsafe_root "$_custom_root"; then
            echo "  refusing to remove unsafe path: $_custom_root" >&2
            # A real install can sit under a deny-listed path; it still holds studio.db, so flag it.
            [ -d "$_custom_root" ] && _set_marker "$_REMOVE_FAILED_FLAG"
            continue
        fi
        if ! _is_studio_root "$_custom_root"; then
            echo "  refusing to remove non-Unsloth path: $_custom_root" >&2
            continue
        fi
        # Flat layout (both variables name one dir): do not take the user-chosen master root with it.
        _crf_canon=$(CDPATH= cd -P -- "$_custom_root" 2>/dev/null && pwd -P) || _crf_canon=""
        [ -n "$_crf_canon" ] || _crf_canon="$_custom_root"
        _crf_master="$_MASTER_ROOT_SAVED"
        if [ -n "$_crf_master" ] && [ "$_crf_canon" = "$_crf_master" ]; then
            echo "  keeping $_custom_root: UNSLOTH_HOME and the Studio root name the same" >&2
            echo "  directory, so removing it would take whatever else you keep there." >&2
            echo "  Delete it by hand once you have checked what is in it." >&2
            _set_marker "$_REMOVE_FAILED_FLAG"
            unset _crf_canon _crf_master
            continue
        fi
        unset _crf_canon _crf_master
        _remove_root_recording_db "$_custom_root"
        # Older builds put sd.cpp beside the root; require our owner marker, since a git
        # clone has the same name.
        _custom_sd_cpp="$(dirname "$_custom_root")/stable-diffusion.cpp"
        if _is_unsafe_root "$_custom_sd_cpp"; then
            echo "  refusing to remove unsafe path: $_custom_sd_cpp" >&2
        elif [ -e "$_custom_sd_cpp" ] && [ ! -f "$_custom_sd_cpp/.unsloth-studio-owned" ]; then
            echo "  keeping sd.cpp without Unsloth owner marker: $_custom_sd_cpp" >&2
        else
            _remove_path "$_custom_sd_cpp"
        fi
    done
    # The lexical parent too: a symlinked home has its old sd.cpp tree beside the link.
    _custom_studio_roots lexical 2>/dev/null | while IFS= read -r _lex_root; do
        [ -n "$_lex_root" ] || continue
        # Same ownership check, or a mistyped UNSLOTH_STUDIO_HOME could take someone's marked sd.cpp.
        _is_studio_root "$_lex_root" || continue
        _lex_sd_cpp="$(dirname "$_lex_root")/stable-diffusion.cpp"
        [ -f "$_lex_sd_cpp/.unsloth-studio-owned" ] || continue
        # The deny list is string-based, so check the resolved path; the rm stays lexical.
        # shellcheck disable=SC1007
        _lex_sd_canon=$(CDPATH= cd -P -- "$_lex_sd_cpp" 2>/dev/null && pwd -P)
        [ -n "$_lex_sd_canon" ] || _lex_sd_canon="$_lex_sd_cpp"
        if _is_unsafe_root "$_lex_sd_cpp" || _is_unsafe_root "$_lex_sd_canon"; then
            echo "  refusing to remove unsafe path: $_lex_sd_cpp" >&2
        else
            _remove_path "$_lex_sd_cpp"
        fi
    done
    # The master root's own children are marker-gated and deny-listed: <master> is user-chosen.
    _mr_root="$_MASTER_ROOT_SAVED"
    if [ -n "$_mr_root" ]; then
        if _is_unsafe_root "$_mr_root"; then
            echo "  refusing to remove unsafe path: $_mr_root" >&2
        else
            for _mr_child in llama.cpp node whisper.cpp audio.cpp stable-diffusion.cpp; do
                _mr_path="$_mr_root/$_mr_child"
                if _is_unsafe_root "$_mr_path"; then
                    echo "  refusing to remove unsafe path: $_mr_path" >&2
                # -L too: -e is false for a dangling link, which in a user-chosen root is theirs.
                elif { [ -e "$_mr_path" ] || [ -L "$_mr_path" ]; } \
                    && [ ! -f "$_mr_path/.unsloth-studio-owned" ]; then
                    echo "  keeping $_mr_child without Unsloth owner marker: $_mr_path" >&2
                else
                    _remove_path "$_mr_path"
                fi
            done
            for _mr_lock in .llama.cpp.install.lock .node.install.lock \
                    .whisper.cpp.install.lock .audio.cpp.install.lock .sd.cpp.install.lock; do
                _remove_lock_file "$_mr_root/$_mr_lock"
            done
            # Shared .staging is pruned only when empty; rmdir, not a recursive delete.
            rmdir "$_mr_root/.staging" 2>/dev/null || true
            # The exact shape prebuilt_core.py leaves: a component lock name, ".stale.", and the
            # pid. A bare .*.install.lock.stale.* also matched .backup.install.lock.stale.copy.
            for _mr_lock in .llama.cpp.install.lock .node.install.lock \
                    .whisper.cpp.install.lock .audio.cpp.install.lock .sd.cpp.install.lock; do
                for _mr_stale in "$_mr_root/$_mr_lock".stale.*; do
                    case "${_mr_stale##*.stale.}" in
                        ''|*[!0-9]*) continue ;;
                    esac
                    { [ -e "$_mr_stale" ] || [ -L "$_mr_stale" ]; } && _remove_lock_file "$_mr_stale"
                done
            done
            rmdir "$_mr_root" 2>/dev/null || true
        fi
    fi
    # end master-root children
    # -e OR -L: _remove_path also unlinks dangling links, so the gate must see them.
    if { [ -e "$HOME/.unsloth/studio" ] || [ -L "$HOME/.unsloth/studio" ]; } \
       && ! _is_studio_root "$HOME/.unsloth/studio" managed; then
        echo "  refusing to remove non-Unsloth path: $HOME/.unsloth/studio" >&2
        # Unlike a refused custom root, a studio.db in our default path is chat history.
        if [ -f "$HOME/.unsloth/studio/studio.db" ]; then
            _set_marker "$_DB_KEPT_FLAG"
        fi
    else
        _remove_root_recording_db "$HOME/.unsloth/studio"
    fi
    # Shared llama.cpp build + cache, siblings of studio in default mode (deleting studio misses
    # them). No-op in env/custom mode and when absent. A user-set UNSLOTH_LLAMA_CPP_PATH is kept.
    _remove_path "$HOME/.unsloth/llama.cpp"
    # A user's own stable-diffusion.cpp checkout may sit here, so require our owner marker.
    _default_sd_cpp="$HOME/.unsloth/stable-diffusion.cpp"
    if [ -e "$_default_sd_cpp" ] && [ ! -f "$_default_sd_cpp/.unsloth-studio-owned" ]; then
        echo "  keeping sd.cpp without Unsloth owner marker: $_default_sd_cpp" >&2
    else
        _remove_path "$_default_sd_cpp"
    fi
    _remove_path "$HOME/.unsloth/.cache"
    _remove_path "$HOME/.unsloth/node"
    # An interrupted llama.cpp install leaves .staging, which blocks the rmdir below.
    _remove_path "$HOME/.unsloth/.staging"
    _remove_path "$HOME/.unsloth/whisper.cpp"
    # The audio.cpp installer always marks its tree, so an unmarked one is the user's build.
    _default_audio_cpp="$HOME/.unsloth/audio.cpp"
    if { [ -e "$_default_audio_cpp" ] || [ -L "$_default_audio_cpp" ]; } \
        && [ ! -f "$_default_audio_cpp/.unsloth-studio-owned" ]; then
        echo "  keeping audio.cpp without Unsloth owner marker: $_default_audio_cpp" >&2
    else
        _remove_path "$_default_audio_cpp"
    fi
    # A stray install lock keeps ~/.unsloth from being pruned below.
    _remove_lock_file "$HOME/.unsloth/.llama.cpp.install.lock"
    _remove_lock_file "$HOME/.unsloth/.node.install.lock"
    _remove_lock_file "$HOME/.unsloth/.whisper.cpp.install.lock"
    _remove_lock_file "$HOME/.unsloth/.audio.cpp.install.lock"
    # audio.cpp model links live beside the HF hub cache; the cache itself stays.
    _hf_hub="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}}"
    _remove_path "$(dirname "$_hf_hub")/unsloth-audiocpp-links"
    # Per-uid audio.cpp scratch home; only a real directory this user owns.
    _audiocpp_home="${TMPDIR:-/tmp}"
    _audiocpp_home="${_audiocpp_home%/}/unsloth-audiocpp-home-$(id -u 2>/dev/null)"
    if [ -d "$_audiocpp_home" ] && [ ! -L "$_audiocpp_home" ] && [ -O "$_audiocpp_home" ]; then
        _remove_path "$_audiocpp_home"
    fi
    # Taking over an abandoned lock renames it to .stale.<pid> before unlinking
    # (install_node_prebuilt.py); a crash between the two strands the rename, and a stranded one
    # blocks the rmdir below. Unmatched globs stay literal, hence the existence test.
    # Same shape restriction as the master-root sweep above: the help text promises that
    # anything else kept under ~/.unsloth is left in place, and .backup.install.lock.stale.copy
    # is dotted too.
    for _lock in .llama.cpp.install.lock .node.install.lock \
            .whisper.cpp.install.lock .audio.cpp.install.lock .sd.cpp.install.lock; do
        for _stale in "$HOME/.unsloth/$_lock".stale.*; do
            case "${_stale##*.stale.}" in
                ''|*[!0-9]*) continue ;;
            esac
            { [ -e "$_stale" ] || [ -L "$_stale" ]; } && _remove_lock_file "$_stale"
        done
    done
    _remove_path "$HOME/.unsloth/librocdxg"
    _remove_path "$HOME/.unsloth/rocm-smoketest"
    rmdir "$HOME/.unsloth" 2>/dev/null || true
    _remove_path "$HOME/.local/share/unsloth"
    _remove_cli_shim

    echo "Removing desktop shortcut and launcher lock..."
    _desktop_link="$HOME/Desktop/Unsloth Studio"
    if [ -L "$_desktop_link" ] || [ ! -e "$_desktop_link" ]; then
        _remove_path "$_desktop_link"
    else
        echo "  refusing to remove non-symlink Desktop path: $_desktop_link" >&2
    fi
    _remove_path "$HOME/Desktop/unsloth-studio.desktop"
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
            # While the packaged app is installed, do not reset its WebView data.
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
                # defaults, not rm: cfprefsd rewrites the plist from memory.
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
                # Remove only THIS distro's WSL shortcuts. Test powershell.exe can execute:
                # `command -v` succeeds even with interop off.
                _wsl_distro="${WSL_DISTRO_NAME:-}"
                _ps_ran=0
                if command -v powershell.exe >/dev/null 2>&1 && \
                   powershell.exe -NoProfile -Command "exit 0" >/dev/null 2>&1; then
                    _ps_ran=1
                    # A -Command string does not receive trailing tokens as $args, so inject the distro.
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
                # Keep the shared icon while any shortcut uses it.
                # Reciprocal of uninstall.ps1's _RemoveDataDirKeepingWslIcon.
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
                                    _lnk="$_scdir/Unsloth Studio (WSL - ${_wsl_distro}).lnk"
                                    [ -e "$_lnk" ] && rm -f "$_lnk" 2>/dev/null && echo "  removed: $_lnk" || true
                                else
                                    for _lnk in "$_scdir"/"Unsloth Studio (WSL"*.lnk; do
                                        [ -e "$_lnk" ] && rm -f "$_lnk" 2>/dev/null && echo "  removed: $_lnk" || true
                                    done
                                fi
                            done
                            _drop_shared_icon_if_unused "$_udir"
                        done
                    done
                fi
                # System ROCm is kept unless UNSLOTH_UNINSTALL_ROCM=1.
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
            _bid="ai.unsloth.studio"
            echo "Removing WebView caches and app data ($_bid)..."
            _remove_path "$(_xdg_dir "${XDG_DATA_HOME:-}" "$HOME/.local/share")/$_bid"
            _remove_path "$(_xdg_dir "${XDG_CACHE_HOME:-}" "$HOME/.cache")/$_bid"
            _remove_path "$(_xdg_dir "${XDG_CONFIG_HOME:-}" "$HOME/.config")/$_bid"
            _remove_path "$(_xdg_dir "${XDG_STATE_HOME:-}" "$HOME/.local/state")/$_bid"
            echo "Removing Linux .desktop entry..."
            _remove_path "$HOME/.local/share/applications/unsloth-studio.desktop"
            # tauri-plugin-deep-link rewrites this on every launch and honours XDG_DATA_HOME.
            _un_appdir="$(_xdg_dir "${XDG_DATA_HOME:-}" "$HOME/.local/share")/applications"
            _remove_path "$_un_appdir/unsloth-studio-handler.desktop"
            if [ "$_un_appdir" != "$HOME/.local/share/applications" ]; then
                _remove_path "$HOME/.local/share/applications/unsloth-studio-handler.desktop"
            fi
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
        echo "Note: some paths could not be removed (see 'could not remove:' above), so the"
        echo "      signed-in session and local chat history may still be on disk. Remove"
        echo "      those paths by hand to clear them."
    elif _marker_set "$_DB_REMOVED_FLAG"; then
        # A bare run never discovers an env-mode root, so do not claim its database is gone.
        echo "Note: this also removed the app's WebView data and the studio.db it found, so"
        echo "      the desktop app's session and the chat history in the install(s) removed"
        echo "      above are gone."
    else
        echo "Note: this also removed the app's WebView data, so the desktop app's session is"
        echo "      gone. A browser session is not affected: its tokens live in the same"
        echo "      localStorage as the API keys below."
        if _marker_set "$_DB_KEPT_FLAG"; then
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
    _uv_print_leftover_notes
    if [ -z "${UNSLOTH_STUDIO_HOME:-}" ] && [ -z "${STUDIO_HOME:-}" ]; then
        echo ""
        echo "If you installed Unsloth Studio with UNSLOTH_STUDIO_HOME or STUDIO_HOME"
        echo "pointing at a custom directory, re-run this script with the same variable"
        echo "set to also remove that install tree, e.g.:"
        echo "  UNSLOTH_STUDIO_HOME=/your/path sh -c \"\$(curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh)\""
    fi
}

# Parse the whole script first so a truncated download is inert.
{
    _unsloth_uninstall_main "$@"
}
