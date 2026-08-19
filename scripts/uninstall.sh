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

# sd.cpp roots whose sd-server has to be stopped: the default
# $HOME/.unsloth/stable-diffusion.cpp, plus for each custom root both
# <root>/stable-diffusion.cpp, where the install now lives, and the legacy
# <parent>/stable-diffusion.cpp sibling an older build wrote. The nested one matters most: a
# resident sd-server survives unlinking its binary, and the custom root is removed wholesale
# below, so without it the tree goes and the server keeps running.
# The owner marker gates the paths that SURVIVE when unowned (the default and the sibling), so an
# unrelated checkout there keeps its server. It does not gate the nested path of a root this run
# deletes: the current-root finder can select an unmarked binary there, and deleting the tree out
# from under a live server is exactly what leaves it holding its port.
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
# and the lexical ones. They differ only when the Unsloth home is itself a symlink, and there the
# lexical form is the one the old `dirname "$UNSLOTH_STUDIO_HOME"` produced, so canonicalizing
# first looked beside the link's target and missed the tree entirely. Every use is gated on the
# owner marker, which is what keeps an unrelated checkout at either path safe.
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
        # No procps (install.sh never requires it): the PID sweep above is all we have.
        # A live app's WebView helpers re-create the profile right after the delete, so
        # the removal is incomplete and the summary must not claim otherwise.
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
        # `unsloth studio` (default port) + `-p N` + `--port N` forms, all
        # anchored on the install root's venv path.
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

    # Native diffusion servers (sd-server / sd-cli) survive unlinking their binary,
    # so stop the ones under an owned sd.cpp root before those trees are removed.
    _stop_owned_sd_cpp_processes TERM
    sleep 0.5
    _stop_owned_sd_cpp_processes KILL

    # The app's WebView helpers re-create the caches removed below, so it has to die here.
    # -x is exact, so the "unsloth" CLI shim never matches. -u takes the owner of the $HOME
    # being cleared, not the caller (macOS sudo keeps HOME), so a root run spares other users;
    # an unknown owner skips rather than signalling everyone. Numeric uid and signal-first
    # suit BSD pkill, which reads the signal from argv[1] only.
    _studio_uid=$(_home_uid)
    if [ -n "$_studio_uid" ]; then
        pkill -TERM -x -u "$_studio_uid" unsloth-studio 2>/dev/null || true
        sleep 0.5
        pkill -KILL -x -u "$_studio_uid" unsloth-studio 2>/dev/null || true
    fi
}

# Summary state in files, not variables: custom roots are removed inside a pipeline
# subshell, where an assignment would never reach the summary.
#   remove-failed  an rm failed, or a root was skipped while still holding data
#   db-removed     a removed install root actually held studio.db
# studio.db holds chat_threads/chat_messages (backend/storage/studio_db.py via studio_root()),
# not the provider API keys: providers_db.py keeps those in the browser's localStorage only.
# It sits under the install root, so an env-mode install keeps it in a custom root a bare run
# cannot discover; claim the history is gone only if a database was really deleted.
# mktemp -d, not a $TMPDIR name: private (0700) and unpredictable, so the markers cannot
# collide with, or be pre-created by, anything else.
_MARKER_DIR=$(mktemp -d 2>/dev/null || true)
_REMOVE_FAILED_FLAG=""
_DB_REMOVED_FLAG=""
if [ -n "$_MARKER_DIR" ] && [ -d "$_MARKER_DIR" ]; then
    _REMOVE_FAILED_FLAG="$_MARKER_DIR/remove-failed"
    _DB_REMOVED_FLAG="$_MARKER_DIR/db-removed"
fi

# `printf`, never `: > "$f"`: `:` is a POSIX special builtin, so a redirection error on it
# kills a non-interactive shell outright (dash 2, busybox ash 1) and `|| true` does not stop
# it. printf is a regular builtin, so the same failure is just a nonzero status.
_set_marker() {
    [ -n "$1" ] || return 0
    printf '' > "$1" 2>/dev/null || true
    return 0
}
_marker_set() { [ -n "$1" ] && [ -f "$1" ]; }
# No marker storage means no record of what failed, so the summary must not claim success.
# Re-checked, not trusted from startup: the directory can vanish or lose write access mid-run,
# after which _set_marker silently drops every failure while the pathname still looks fine.
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

# Remove an install root and record whether its studio.db really went with it.
# The check runs on the RESOLVED path: a relocated install (~/.unsloth/studio a symlink to
# another disk) passes `-f "$root/studio.db"` through the link, but `rm -rf` unlinks only the
# link, and afterwards the path stops resolving and reads as absent either way.
# Verifying rather than chasing the link is deliberate: following a symlink out of the
# expected location to `rm -rf` its target is what the deny lists exist to prevent.
_remove_root_recording_db() {
    _rrd_root="$1"
    # shellcheck disable=SC1007
    _rrd_real=$(CDPATH= cd -P -- "$_rrd_root" 2>/dev/null && pwd -P) || _rrd_real=""
    [ -n "$_rrd_real" ] || _rrd_real="$_rrd_root"
    _rrd_had_db=0
    _rrd_db="$_rrd_real/studio.db"
    if [ -f "$_rrd_db" ]; then
        _rrd_had_db=1
        # The db itself can be a symlink out of the tree: -f follows it but the rm below
        # unlinks only the link, so track where the bytes are. readlink without -f: BSD
        # gained -f in macOS 12.3, so use the raw link text plus cd -P, portable to both.
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

# $1 override, $2 default. A relative override is invalid per XDG and dropped by dirs (which
# Tauri resolves through), so honouring one would spare the real data and rm -rf under our cwd.
_xdg_dir() {
    case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$2" ;; esac
}

# Accept as Unsloth root only if Unsloth sentinels exist (matches install.sh's
# env-mode ownership guard at install.sh:1358-1361). A bare unsloth_studio/
# directory is NOT enough -- require the install-time owner marker so a user
# directory that happens to contain a folder named "unsloth_studio" is safe.
_is_studio_root() {
    _r="$1"
    [ -n "$_r" ] || return 1
    [ -f "$_r/share/studio.conf" ] && return 0
    [ -f "$_r/unsloth_studio/.unsloth-studio-owned" ] && return 0
    if [ -L "$_r/bin/unsloth" ]; then
        _t=$(readlink "$_r/bin/unsloth" 2>/dev/null || true)
        case "$_t" in *unsloth_studio/bin/unsloth) return 0 ;; esac
    fi
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
# install.sh writes UNSLOTH_EXE='<root>/unsloth_studio/bin/unsloth', so
# the install root is three dirnames up. Prints each discovered non-default
# root on its own line; the caller iterates and de-duplicates.
_custom_studio_roots() {
    # $1 = "lexical": skip the canonicalization (see the legacy sd.cpp sibling below). Reset on
    # every call, so a plain call is never affected by a preceding lexical one.
    _studio_roots_lexical="${1:-}"
    _seen=""
    _emit() {
        _r="$1"
        [ -z "$_r" ] && return 0
        # Tilde expansion (env vars are not subject to it on quoted assignment),
        # matches install.sh's _resolve_studio_destinations. The literal "~/"
        # pattern is intentional; SC2088 is a false positive here.
        # shellcheck disable=SC2088
        case "$_r" in
            "~") _r="$HOME" ;;
            "~/"*) _r="$HOME/${_r#'~/'}" ;;
        esac
        # Canonicalize so syntactic variants ($HOME/../$USER, trailing slash)
        # resolve to the same path and hit the _is_unsafe_root deny list.
        # Skipped for the lexical pass, which exists only to rebuild the path an
        # older build derived with a plain dirname (see _sd_cpp_sibling_bases).
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
    # Mirror install.sh's precedence: UNSLOTH_STUDIO_HOME wins, STUDIO_HOME is
    # ignored when both are set. Otherwise uninstalling install A could also
    # delete install B if the user has STUDIO_HOME left over from B.
    if [ -n "${UNSLOTH_STUDIO_HOME:-}" ]; then
        _emit "$UNSLOTH_STUDIO_HOME"
        _from_conf "$UNSLOTH_STUDIO_HOME/share/studio.conf"
    elif [ -n "${STUDIO_HOME:-}" ]; then
        _emit "$STUDIO_HOME"
        _from_conf "$STUDIO_HOME/share/studio.conf"
    fi
    # Default-mode conf.
    _from_conf "$HOME/.local/share/unsloth/studio.conf"
}

# Remove $HOME/.local/bin/unsloth only if it's an Unsloth-managed symlink.
# Unsloth's install.sh writes this as a symlink into the studio venv
# (install.sh: `ln -sfn "$VENV_DIR/bin/unsloth" "$_shim_path"`). A
# pip-installed `unsloth` CLI is a regular file — leave it alone to avoid
# wiping an unrelated install.
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

# Path of the installed app owning bundle id $1, empty if there is none. A renamed
# bundle or a subdirectory such as "/Applications/AI & ML/Unsloth.app" is a supported
# layout, so match on the identifier rather than on a fixed set of paths.
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

    _remove_systemd_user_service() {
        _sd_unit_dir="$(_xdg_dir "${XDG_CONFIG_HOME:-}" "$HOME/.config")/systemd/user"
        _sd_unit="$_sd_unit_dir/unsloth-studio.service"
        [ -f "$_sd_unit" ] || return 0
        grep -q 'unsloth-studio-managed-systemd' "$_sd_unit" 2>/dev/null || return 0
        if command -v systemctl >/dev/null 2>&1 \
                && systemctl --user show-environment >/dev/null 2>&1; then
            systemctl --user disable --now unsloth-studio.service 2>/dev/null || true
            systemctl --user daemon-reload 2>/dev/null || true
        fi
        _remove_path "$_sd_unit"
        echo "Removed systemd user service (unsloth-studio.service)."
    }
    _remove_systemd_user_service

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
        # Native diffusion (stable-diffusion.cpp) now installs UNDER the custom root, at
        # <root>/stable-diffusion.cpp, so the removal above already took it. Older builds put it
        # BESIDE the root at <parent>/stable-diffusion.cpp (find_sd_cpp_binary derived it from
        # UNSLOTH_STUDIO_HOME.parent), and removing only the root would leave that build behind.
        # Only remove a sibling Unsloth installed: <parent> is a user-chosen dir and
        # "stable-diffusion.cpp" is exactly what `git clone` of the upstream project produces, so
        # require our owner marker (written by install_sd_cpp_prebuilt) before rm, and keep any
        # unowned checkout. A pre-marker Unsloth build is left behind, never a user file deleted.
        # Guard the derived parent path the same way.
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
    # the LINK, and the loop above only saw the canonicalized root, so that tree survived. Marker
    # only, with no "keeping" notice: an unmarked directory at this path is somebody's checkout
    # and the canonical pass has already reported the one it looked at.
    _custom_studio_roots lexical 2>/dev/null | while IFS= read -r _lex_root; do
        [ -n "$_lex_root" ] || continue
        # The same ownership check the canonical loop makes before it touches anything. A stale or
        # mistyped UNSLOTH_STUDIO_HOME still reaches here (the lexical pass has no cd -P to filter
        # a path that is not there), and without this "/parent/typo" would take the marked
        # /parent/stable-diffusion.cpp of somebody else's Unsloth with it.
        _is_studio_root "$_lex_root" || continue
        _lex_sd_cpp="$(dirname "$_lex_root")/stable-diffusion.cpp"
        [ -f "$_lex_sd_cpp/.unsloth-studio-owned" ] || continue
        # The deny list is string-based, so it has to see the RESOLVED path: the lexical form can
        # carry ".." or a symlinked ancestor and slip a protected tree ("/tmp/../usr/...") past it.
        # Canonicalize a copy for the check only; the removal still uses the lexical path.
        # shellcheck disable=SC1007
        _lex_sd_canon=$(CDPATH= cd -P -- "$_lex_sd_cpp" 2>/dev/null && pwd -P)
        [ -n "$_lex_sd_canon" ] || _lex_sd_canon="$_lex_sd_cpp"
        if _is_unsafe_root "$_lex_sd_cpp" || _is_unsafe_root "$_lex_sd_canon"; then
            echo "  refusing to remove unsafe path: $_lex_sd_cpp" >&2
        else
            _remove_path "$_lex_sd_cpp"
        fi
    done
    _remove_root_recording_db "$HOME/.unsloth/studio"
    # Default-mode shared llama.cpp build + cache are siblings of studio (not removed
    # by deleting it). No-op in env/custom mode (they nest under the custom root) and
    # when absent. A user-set UNSLOTH_LLAMA_CPP_PATH is intentionally kept.
    _remove_path "$HOME/.unsloth/llama.cpp"
    # Default-mode native diffusion (stable-diffusion.cpp / sd-cli) build, a sibling of
    # studio like llama.cpp (install_sd_cpp_prebuilt.default_install_dir()). No-op in
    # env/custom mode and when absent. "stable-diffusion.cpp" is exactly what a `git clone` of
    # leejet/stable-diffusion.cpp produces, so a user may keep their own checkout (or point
    # UNSLOTH_SD_CPP_PATH) at this default path; require our owner marker (written by
    # install_sd_cpp_prebuilt) before rm, mirroring the custom-root guard above, so a user's own
    # checkout or a pre-marker Unsloth build is kept rather than deleted.
    _default_sd_cpp="$HOME/.unsloth/stable-diffusion.cpp"
    if [ -e "$_default_sd_cpp" ] && [ ! -f "$_default_sd_cpp/.unsloth-studio-owned" ]; then
        echo "  keeping sd.cpp without Unsloth owner marker: $_default_sd_cpp" >&2
    else
        _remove_path "$_default_sd_cpp"
    fi
    _remove_path "$HOME/.unsloth/.cache"
    # Isolated Node.js runtime (install_node_prebuilt.py), a sibling of studio in
    # default mode. No-op in env/custom mode (nested under the custom root) and absent.
    _remove_path "$HOME/.unsloth/node"
    # llama.cpp atomic-install staging root (install_llama_prebuilt.py .staging).
    # Normally pruned after activate, but an interrupted build can leave it behind;
    # removing it lets the rmdir below succeed. No-op in env/custom mode and absent.
    _remove_path "$HOME/.unsloth/.staging"
    # Managed whisper.cpp dictation engine (install_whisper_prebuilt.py), a sibling
    # of studio in default mode. Only present when a whisper prebuilt matching the
    # pinned llama.cpp build existed at install time, so many installs lack it.
    _remove_path "$HOME/.unsloth/whisper.cpp"
    # Prebuilt install locks. Every prebuilt serializes on
    # <parent>/.<name>.install.lock (prebuilt_core.py:1129), so llama.cpp, node and
    # whisper.cpp each leave one; a stray lock keeps ~/.unsloth from being pruned
    # below. No-op in env/custom mode and when absent.
    _remove_path "$HOME/.unsloth/.llama.cpp.install.lock"
    _remove_path "$HOME/.unsloth/.node.install.lock"
    _remove_path "$HOME/.unsloth/.whisper.cpp.install.lock"
    # Taking over an abandoned lock renames it to .stale.<pid> before unlinking
    # (install_node_prebuilt.py); a crash between the two steps strands the rename,
    # and a stranded one blocks the rmdir below. Unmatched globs stay literal,
    # hence the existence test.
    for _stale in "$HOME"/.unsloth/.*.install.lock.stale.*; do
        [ -e "$_stale" ] && _remove_path "$_stale"
    done
    # ROCm-on-WSL helper artifacts (librocdxg build clone + smoke-test venv). No-op
    # where they don't exist; removing them lets the rmdir below succeed.
    _remove_path "$HOME/.unsloth/librocdxg"
    _remove_path "$HOME/.unsloth/rocm-smoketest"
    # Drop ~/.unsloth only if now empty (rmdir refuses non-empty, so user content is kept).
    rmdir "$HOME/.unsloth" 2>/dev/null || true
    _remove_path "$HOME/.local/share/unsloth"
    # CLI shim: only the symlink Unsloth created, never a pip-installed file.
    _remove_cli_shim

    echo "Removing desktop shortcut and launcher lock..."
    # install.sh creates Desktop/Unsloth Studio as a symlink. If the user has an
    # unrelated regular directory by that name, leave it alone.
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
            # WKWebView data, keyed by bundle id. Created at first launch, not by install.sh.
            # The packaged desktop app shares this bundle id and is the only thing that
            # writes this data; the shell launcher just opens a browser. This script never
            # removes that app, so it must not reset it either.
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
                # install.sh creates per-distro 'Unsloth Studio (WSL - <distro>).lnk'
                # on the Windows Desktop + Start Menu via powershell.exe. Scope removal
                # to THIS distro (passed as $args[0]) so a multi-distro install keeps the
                # other distros' launchers; the TARGET=wsl.exe check still spares a
                # native install's "Unsloth Studio.lnk". Prefer powershell.exe; test it
                # can EXECUTE (`command -v` succeeds even with interop OFF -- .exe then
                # fails "Exec format error", common on systemd-enabled distros).
                _wsl_distro="${WSL_DISTRO_NAME:-}"
                _ps_ran=0
                if command -v powershell.exe >/dev/null 2>&1 && \
                   powershell.exe -NoProfile -Command "exit 0" >/dev/null 2>&1; then
                    _ps_ran=1
                    # Inject the distro into the command: a -Command string does not
                    # receive trailing tokens as $args. WSL distro names are safe to
                    # embed (no quotes/$/backtick).
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
                # Remove $1's shared unsloth.ico only if no Unsloth shortcut (native install
                # or another WSL distro) still uses it, then drop the dir if empty. Reciprocal
                # of uninstall.ps1's _RemoveDataDirKeepingWslIcon (keeps the icon for a
                # surviving WSL shortcut when the native side is removed).
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
                # Fallback when powershell.exe can't run (interop disabled): remove WSL .lnk
                # files via drvfs. The "Unsloth Studio (WSL..." name is WSL-specific, so a
                # native install's "Unsloth Studio.lnk" never matches.
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
            # unsloth:// scheme, so it exists on any machine the app has started on and would
            # be left pointing at a binary we just deleted. Unlike install.sh's own shortcut,
            # it uses Tauri's data_dir(), which honours XDG_DATA_HOME, so check both.
            _un_appdir="$(_xdg_dir "${XDG_DATA_HOME:-}" "$HOME/.local/share")/applications"
            _remove_path "$_un_appdir/unsloth-studio-handler.desktop"
            if [ "$_un_appdir" != "$HOME/.local/share/applications" ]; then
                _remove_path "$HOME/.local/share/applications/unsloth-studio-handler.desktop"
            fi
            # Rebuild mimeinfo.cache wherever an entry was removed, or the stale cache keeps
            # advertising it. Mirrors install.sh:1574 on the way out.
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
        echo "      No studio.db was found, so any chat history in an install root this run"
        echo "      did not see is still on disk."
    fi
    echo "Note: provider API keys are kept in the browser's localStorage, not in studio.db."
    echo "      Unless you ran Unsloth as the desktop app, clear site data for the"
    echo "      http://localhost:<port> origin you used to remove them."
    echo "Note: Hugging Face model cache at ~/.cache/huggingface was left in place."
    echo "Remove it manually with 'rm -rf ~/.cache/huggingface/hub' if desired."
    # Env-mode installs leave no breadcrumb in $HOME, so a custom root can
    # only be located if the user re-exports the variable. Print a hint when
    # neither var is set so the bare `curl | sh` flow doesn't silently miss.
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
