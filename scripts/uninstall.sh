#!/usr/bin/env sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
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
# Stop a detached CUDA llama.cpp build BEFORE deleting its tree: _pkill_studio
# only matches Studio roots, and a live cmake/nvcc under ~/.unsloth/llama.cpp
# would keep burning CPU/thermals, recreate build/ files between the rm and the
# rmdir, and leave a partial tree. TERM first, then KILL after the same grace
# _pkill_studio uses.
if command -v pkill >/dev/null 2>&1; then
    _llama_re=$(_pkill_escape "$HOME/.unsloth/llama.cpp")
    # Signal the whole process GROUP of each match, not just the matching PID:
    # the provisioner cds into the tree before `cmake --build build`, so cmake/
    # make children carry relative argv that no pattern can match, and killing
    # only the wrapper orphans them mid-build. Group kill sweeps the tree; PID
    # kill remains the fallback when pgid is unreadable or shared with init.
    _kill_llama_build() {
        _sig="$1"
        for _pat in "run_llama_build\.sh" "provision_llama_cuda\.sh" "$_llama_re"; do
            for _pid in $(pgrep -f "$_pat" 2>/dev/null); do
                _pgid=$(ps -o pgid= -p "$_pid" 2>/dev/null | tr -d '[:space:]')
                case "$_pgid" in
                    ''|0|1) kill -s "$_sig" "$_pid" 2>/dev/null || true ;;
                    *) kill -s "$_sig" -- "-$_pgid" 2>/dev/null \
                        || kill -s "$_sig" "$_pid" 2>/dev/null || true ;;
                esac
            done
        done
    }
    _kill_llama_build TERM
    sleep 0.5
    _kill_llama_build KILL
fi
# Default-mode shared llama.cpp build + cache are siblings of studio (not removed
# by deleting it). No-op in env/custom mode (they nest under the custom root) and
# when absent. A user-set UNSLOTH_LLAMA_CPP_PATH is intentionally kept.
_remove_path "$HOME/.unsloth/llama.cpp"
# WoA/Spark CUDA-build path artifacts (provision script fetched by setup.sh,
# install.ps1's background-build runner + log, and the persisted shortcut-skip
# marker). No-ops when absent.
_remove_path "$HOME/.unsloth/provision_llama_cuda.sh"
_remove_path "$HOME/.unsloth/run_llama_build.sh"
_remove_path "$HOME/.unsloth/llama_cuda_build.log"
_remove_path "$HOME/.unsloth/.skip-wsl-windows-shortcut"
# Core-install completion stamp (written by setup.sh, checked by install.ps1's
# WSL probes). Must go, or a later reinstall could read a stale success.
_remove_path "$HOME/.unsloth/.install-ok"
_remove_path "$HOME/.unsloth/unsloth-install.sh"
_remove_path "$HOME/.unsloth/.cache"
# Isolated Node.js runtime (install_node_prebuilt.py), a sibling of studio in
# default mode. No-op in env/custom mode (nested under the custom root) and absent.
_remove_path "$HOME/.unsloth/node"
# llama.cpp atomic-install staging root (install_llama_prebuilt.py .staging).
# Normally pruned after activate, but an interrupted build can leave it behind;
# removing it lets the rmdir below succeed. No-op in env/custom mode and absent.
_remove_path "$HOME/.unsloth/.staging"
# llama.cpp install lock (serializes the shared build); a stray one keeps ~/.unsloth
# from being pruned below. No-op in env/custom mode and when absent.
_remove_path "$HOME/.unsloth/.llama.cpp.install.lock"
# ROCm-on-WSL helper artifacts (librocdxg build clone + smoke-test venv). No-op
# where they don't exist; removing them lets the rmdir below succeed.
_remove_path "$HOME/.unsloth/librocdxg"
_remove_path "$HOME/.unsloth/rocm-smoketest"
# Drop ~/.unsloth only if now empty (rmdir refuses non-empty, so user content is kept).
rmdir "$HOME/.unsloth" 2>/dev/null || true
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
                # $env:APPDATA/$distro are PowerShell-side; $_wsl_distro is shell-injected.
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
                    # Remove the WoA WSL-fallback native shim/launcher dir
                    # (%LOCALAPPDATA%\Unsloth) + its PATH entry that install.ps1
                    # created, so a WSL-side bash uninstall is complete. Only when THIS
                    # distro owns the fallback (wsl-distro.txt) -- else uninstalling a
                    # different distro would break the still-installed shim.
                    $ud = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA "Unsloth" } else { $null };
                    $owner = $null;
                    if ($ud) { $of = Join-Path $ud "wsl-distro.txt"; if (Test-Path -LiteralPath $of) { $owner = (Get-Content -LiteralPath $of | Select-Object -First 1).Trim() } }
                    if ($ud -and ((-not $owner) -or (-not $distro) -or ($owner -ieq $distro))) {
                        $shim = (Join-Path $ud "bin").TrimEnd("\","/");
                        $up = [Environment]::GetEnvironmentVariable("Path","User");
                        if ($up) { [Environment]::SetEnvironmentVariable("Path", (($up -split ";" | Where-Object { $_ -and ($_.TrimEnd("\","/") -ine $shim) }) -join ";"), "User") }
                        # The WoA-fallback shortcuts target powershell.exe + launch-studio-wsl.ps1
                        # (not wsl.exe), so the sweep above keeps them; remove them here before
                        # their launcher dir is deleted or they would dangle.
                        foreach ($d in $dirs) {
                            if (-not $d -or -not (Test-Path -LiteralPath $d)) { continue }
                            $l = Join-Path $d "Unsloth Studio.lnk";
                            if (Test-Path -LiteralPath $l) {
                                try { $sc2 = $ws.CreateShortcut($l); if ($sc2.Arguments -match "launch-studio-wsl\.ps1") { Remove-Item -LiteralPath $l -Force -ErrorAction SilentlyContinue } } catch { }
                            }
                        }
                        if (Test-Path -LiteralPath $ud) { Remove-Item -LiteralPath $ud -Recurse -Force -ErrorAction SilentlyContinue }
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
                    }
                    # install.sh also writes the WSL shortcut icon to the Windows
                    # profile (%USERPROFILE%\.unsloth\unsloth.ico) because the WoA
                    # icon broker cannot read AppData\Local; clean it the same way.
                    if (-not [string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
                        $pIconDir = Join-Path $env:USERPROFILE ".unsloth";
                        $pIco = Join-Path $pIconDir "unsloth.ico";
                        if ((-not $iconInUse) -and (Test-Path -LiteralPath $pIco)) { Remove-Item -LiteralPath $pIco -Force -ErrorAction SilentlyContinue }
                        if ((Test-Path -LiteralPath $pIconDir) -and -not (Get-ChildItem -LiteralPath $pIconDir -Force -ErrorAction SilentlyContinue)) { Remove-Item -LiteralPath $pIconDir -Recurse -Force -ErrorAction SilentlyContinue }
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
                    # install.sh also writes the icon to the Windows profile
                    # (%USERPROFILE%\.unsloth) for the WoA icon broker.
                    [ -f "$_du/.unsloth/unsloth.ico" ] && rm -f "$_du/.unsloth/unsloth.ico" 2>/dev/null || true
                    [ -d "$_du/.unsloth" ] && rmdir "$_du/.unsloth" 2>/dev/null || true
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
    echo "  UNSLOTH_STUDIO_HOME=/your/path sh -c \"\$(curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/scripts/uninstall.sh)\""
fi
