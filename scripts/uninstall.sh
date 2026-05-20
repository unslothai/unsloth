#!/usr/bin/env sh
# Unsloth Studio uninstaller (macOS / Linux / WSL).
# Stops running servers and removes install dir, launcher data,
# CLI shim, desktop shortcut, .app bundle, and Launch Services entry.
# Honors custom roots set via UNSLOTH_STUDIO_HOME / STUDIO_HOME at
# install time (read back from studio.conf).
#
# Usage: curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh | sh

set -e

# Stop a Studio server via its PID file (written by install.sh's _spawn_terminal).
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

_pkill_studio() {
    # Prefer PID files written by _spawn_terminal so we only touch our own installs.
    for _data_dir in "$HOME/.local/share/unsloth" $(_custom_studio_data_dirs); do
        [ -d "$_data_dir" ] || continue
        for _pf in "$_data_dir"/studio-*.pid; do
            [ -f "$_pf" ] && _kill_pid_file "$_pf"
        done
    done

    command -v pkill >/dev/null 2>&1 || return 0

    # Scope fallback patterns to the install roots we are removing so a
    # different Studio install (different UNSLOTH_STUDIO_HOME) is not touched.
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
}

_remove_path() {
    _p="$1"
    if [ -e "$_p" ] || [ -L "$_p" ]; then
        rm -rf "$_p" 2>/dev/null && echo "  removed: $_p" || echo "  could not remove: $_p" >&2
    fi
}

# Accept as Studio root only if Studio sentinels exist (matches install.sh's
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
        # shellcheck disable=SC1007
        _canon=$(CDPATH= cd -P -- "$_r" 2>/dev/null && pwd -P)
        [ -n "$_canon" ] && _r="$_canon"
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

# Remove $HOME/.local/bin/unsloth only if it's a Studio-managed symlink.
# Studio's install.sh writes this as a symlink into the studio venv
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
        continue
    fi
    if ! _is_studio_root "$_custom_root"; then
        echo "  refusing to remove non-Studio path: $_custom_root" >&2
        continue
    fi
    _remove_path "$_custom_root"
done
_remove_path "$HOME/.unsloth/studio"
_remove_path "$HOME/.local/share/unsloth"
# CLI shim: only the symlink Studio created, never a pip-installed file.
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
        ;;
    Linux)
        if [ "$_is_wsl" = "1" ]; then
            echo "Removing WSL Windows-side shortcuts..."
            # install.sh creates 'Unsloth Studio.lnk' on the Windows Desktop and
            # Start Menu Programs folder via powershell.exe; mirror that path.
            if command -v powershell.exe >/dev/null 2>&1; then
                # shellcheck disable=SC2016
                # $env:APPDATA is a PowerShell expansion; intentionally literal at shell level.
                powershell.exe -NoProfile -Command '
                    $names = @("Desktop","StartMenu");
                    $dirs = @(
                        [Environment]::GetFolderPath("Desktop"),
                        (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs")
                    );
                    foreach ($d in $dirs) {
                        if (-not $d) { continue }
                        $p = Join-Path $d "Unsloth Studio.lnk";
                        if (Test-Path -LiteralPath $p) { Remove-Item -LiteralPath $p -Force }
                    }' >/dev/null 2>&1 || true
            fi
        fi
        echo "Removing Linux .desktop entry..."
        _remove_path "$HOME/.local/share/applications/unsloth-studio.desktop"
        if command -v update-desktop-database >/dev/null 2>&1; then
            update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
        fi
        ;;
esac

echo ""
echo "Unsloth Studio uninstalled."
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
    echo "  UNSLOTH_STUDIO_HOME=/your/path sh uninstall.sh"
fi
